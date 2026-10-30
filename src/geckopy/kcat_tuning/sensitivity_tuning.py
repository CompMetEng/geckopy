"""Bump the most-limiting kcats until the model can grow fast enough.

After ``make_ec_model`` + ``apply_kcat_constraints``, the model
often can't reach the experimental growth rate because one or two
kcat values are unrealistically low (BRENDA's database has many
small or wrong entries). The right long-term fix is manual
curation, but as a quick automated workaround
``sensitivity_tuning`` does the following loop:

1. Solve the model.
2. If we reached the target growth rate, stop.
3. Otherwise, find the enzyme whose usage reaction carries the
   most flux (the bottleneck), and multiply its kcat by
   ``fold_change``.
4. Re-solve and go to 1.

The result is a record of which kcats were bumped and by how
much, so a curator can review the choices. Default target growth
rate is ``adapter.params.gr_exp`` (the experimental growth rate
from the adapter config).

Ported from GECKO MATLAB:
src/geckomat/kcat_sensitivity_analysis/sensitivityTuning.m.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ..ec_model.constants import (
    POOL_ID,
    PROT_PREFIX,
    USAGE_PREFIX,
)
from ..ec_model.pipeline.apply_kcat import apply_kcat_constraints
from ..ec_model.pipeline.populate_ec import split_light_rxn_id
from .corrections import Correction, corrections as build_corrections

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)


_TUNING_SOURCE = "sensitivityTuning"
# Growth is considered stalled when an iteration improves it by no more
# than this (absolute, 1/h). Using a tolerance instead of exact equality
# avoids spinning on sub-numerical-noise increases.
_GROWTH_STALL_TOL = 1e-9


def _num2str(value: float) -> str:
    """A scalar as MATLAB's ``num2str`` prints it: integers as such, other
    values with ``max(floor(log10(abs(x))) + 5, 5)`` significant digits."""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    digits = max(math.floor(math.log10(abs(value))) + 5, 5)
    return f"{value:.{digits}g}"


@dataclass
class TunedKcatsResult:
    """Outcome of a sensitivity-tuning run.

    Attributes
    ----------
    rxns
        IDs of the reactions whose kcat was bumped, in model order (the
        order of ``model.reactions``), as ``sensitivityTuning.m`` lists them.
    rxn_names
        Reaction names parallel to ``rxns`` (falls back to the
        reaction ID when it has no name).
    enzymes
        For each tuned reaction, the catalysing enzyme ID(s),
        ``;``-joined when more than one enzyme is involved.
    old_kcat
        kcat value of the input model, parallel to ``rxns``. A reaction that
        an earlier run had already tuned reports that run's value.
    new_kcat
        kcat value after tuning, parallel to ``rxns``.
    source
        kcat provenance (``model.ec.source``) of the input model,
        parallel to ``rxns``.
    """

    rxns: list[str] = field(default_factory=list)
    rxn_names: list[str] = field(default_factory=list)
    enzymes: list[str] = field(default_factory=list)
    old_kcat: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    new_kcat: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    source: list[str] = field(default_factory=list)

    def corrections(
        self,
        *,
        leverage: Optional[np.ndarray] = None,
        ec_codes: Optional[Sequence[str]] = None,
        rel_tol: float = 0.02,
    ) -> list["Correction"]:
        """The changes this run makes, most consequential first.

        Unlike a screen-backed result, sensitivity_tuning has no
        leverage of its own to offer -- every bump was, by
        construction, the bottleneck at the time it was made -- so
        without ``leverage`` rows fall back to ordering by fold change.
        ``rxn_names`` already carries resolved names, so there is no
        need for ``annotate_from_model`` here.
        """
        return build_corrections(
            self.new_kcat, self.old_kcat, self.rxns,
            sources=self.source, names=self.rxn_names, ec_codes=ec_codes,
            leverage=leverage, rel_tol=rel_tol,
        )


def sensitivity_tuning(
    model: "EcModel",
    *,
    desired_growth_rate: Optional[float] = None,
    fold_change: float = 10.0,
    prot_to_ignore: Optional[list[str]] = None,
    max_iterations: int = 1000,
    bio_rxn: Optional[str] = None,
    verbose: bool = True,
) -> TunedKcatsResult:
    """Iteratively bump the most-limiting kcat by ``fold_change`` until
    the model can reach ``desired_growth_rate``.

    Each iteration:

    1. Solves the LP at the current kcat values.
    2. If growth didn't increase since the last iteration, warns and
       breaks (likely a substrate / pool constraint, not a kcat one).
    3. If growth reached the target, returns.
    4. Otherwise picks the ``usage_prot_*`` reaction with the highest
       flux (= most-produced enzyme), excluding any whose UniProt ID
       is in ``prot_to_ignore``.
    5. Finds the catalysed reaction consuming the most of that
       enzyme's pseudometabolite and multiplies that reaction's
       ``ec.kcat`` by ``fold_change``. A tuned kcat has no upper limit.
    6. Annotates ``ec.notes`` with the original kcat (printed as MATLAB's
       ``num2str`` does) and source, unless the source is already
       ``"sensitivityTuning"``.
    7. Re-applies kcat constraints for that single reaction.

    On a gecko-light model each iteration instead picks the reaction with
    the most flux through the protein pool, multiplies the kcat of all its
    isozyme rows by ``fold_change`` and re-applies the constraint of that
    reaction; ``ec.notes`` and ``ec.source`` are left as they are.

    Ported from GECKO MATLAB:
    src/geckomat/kcat_sensitivity_analysis/sensitivityTuning.m.

    Parameters
    ----------
    model
        EcModel with the protein pool / usage rxn machinery
        installed and a populated ``ec.kcat``. Mutated in place.
    desired_growth_rate
        Target growth rate. Defaults to
        ``model.adapter.params.gr_exp``.
    fold_change
        Per-iteration kcat multiplier.
    prot_to_ignore
        UniProt IDs whose usage rxns are excluded from being picked
        as the limiting enzyme, matched without regard to case. An ID with
        no usage reaction raises ``ValueError`` on a full model; on a
        gecko-light model it is ignored.
    max_iterations
        Hard cap on tuning iterations (``sensitivityTuning.m`` has none). A
        backstop against a target growth rate that is unreachable by
        raising kcats alone.
    bio_rxn
        Biomass reaction id to set as the objective. Defaults to
        ``params.bio_rxn`` from the adapter when one is attached; if
        neither is available the model's current objective is used.
    verbose
        Whether per-iteration progress is logged at INFO.

    Returns
    -------
    TunedKcatsResult

    Raises
    ------
    ValueError
        If the initial LP is infeasible, if ``desired_growth_rate`` is
        needed but no adapter / value is available, or if a
        ``prot_to_ignore`` ID has no usage reaction (full model).
    """
    light = bool(model.ec.gecko_light)

    if desired_growth_rate is None:
        from ..adapter import resolve_adapter
        adapter = resolve_adapter(
            model,
            purpose="sensitivity_tuning needs `desired_growth_rate` "
            "(pass it explicitly, or rely on params.gr_exp from the adapter)",
        )
        if adapter.params.gr_exp is None:
            raise ValueError(
                "desired_growth_rate not provided and "
                "model.adapter.params.gr_exp is unset."
            )
        desired_growth_rate = float(adapter.params.gr_exp)

    if prot_to_ignore is None:
        prot_to_ignore = []

    bio_rxn_id = bio_rxn
    if bio_rxn_id is None and model.adapter is not None:
        bio_rxn_id = model.adapter.params.bio_rxn
    if bio_rxn_id:
        model.objective = bio_rxn_id

    # slim_optimize does not read the primal values, which Gurobi does not hold
    # for an infeasible LP that it presolved (reading them raises GurobiError).
    model.slim_optimize(error_value=float("nan"))
    if model.solver.status != "optimal":
        raise ValueError(
            "FBA of the input model is not optimal (status "
            f"{model.solver.status!r}). Reduce the protein "
            "pool constraint with set_prot_pool_size and/or check the "
            "exchange constraints."
        )

    # The result reports kcat and source as the input model has them.
    input_kcat = np.array(model.ec.kcat, dtype=float)
    input_source = list(model.ec.source)

    usage_rxn_ids = [
        r.id for r in model.reactions
        if r.id.startswith(USAGE_PREFIX)
    ]
    if light:
        ignore_usage_ids: set[str] = set()
        ignore_rxn_ids = _light_rxns_of_enzymes(model, prot_to_ignore)
    else:
        ignore_usage_ids = _usage_ids_to_ignore(usage_rxn_ids, prot_to_ignore)
        ignore_rxn_ids = set()

    last_growth = float("nan")
    # Indices into model.ec.rxns; on a light model the first isozyme row of
    # each tuned reaction.
    tuned_ec_indices: list[int] = []
    iteration = 1

    while iteration <= max_iterations:
        sol = model.optimize()
        if sol.status != "optimal":
            logger.warning(
                "sensitivity_tuning: the model has no optimal solution with the "
                "increased kcats (status %s); tuning stops.", sol.status,
            )
            break
        growth = float(sol.objective_value)

        if not np.isnan(last_growth) and growth <= last_growth + _GROWTH_STALL_TOL:
            logger.warning(
                "sensitivity_tuning: growth did not increase from %g; "
                "uptake or pool constraints may be limiting.", growth,
            )
            break
        last_growth = growth
        if verbose:
            logger.info(
                "sensitivity_tuning: iteration %d, growth = %g",
                iteration, growth,
            )
        if growth >= desired_growth_rate:
            break

        iteration += 1

        if light:
            step = _light_step(model, sol, ignore_rxn_ids)
            if step is None:
                logger.warning(
                    "sensitivity_tuning: no reaction draws on the protein "
                    "pool; cannot continue tuning."
                )
                break
            ec_rows = step
            for i in ec_rows:
                model.ec.kcat[i] = float(model.ec.kcat[i]) * fold_change
            apply_kcat_constraints(model, update_rxns=[model.ec.rxns[ec_rows[0]]])
            if ec_rows[0] not in tuned_ec_indices:
                tuned_ec_indices.append(ec_rows[0])
            continue

        # Pick usage rxn with max flux (= most produced enzyme),
        # excluding any in prot_to_ignore.
        best_usage_id: Optional[str] = None
        best_usage_flux = -np.inf
        for uid in usage_rxn_ids:
            if uid in ignore_usage_ids:
                continue
            f = float(sol.fluxes.get(uid, 0.0))
            if f > best_usage_flux:
                best_usage_flux = f
                best_usage_id = uid

        if best_usage_id is None or best_usage_flux <= 0:
            logger.warning(
                "sensitivity_tuning: no usage rxn carries flux; cannot "
                "continue tuning."
            )
            break

        # The enzyme metabolite the usage rxn produces.
        enzyme_id = best_usage_id[len(USAGE_PREFIX):]
        prot_met_id = f"{PROT_PREFIX}{enzyme_id}"
        try:
            prot_met = model.metabolites.get_by_id(prot_met_id)
        except KeyError:
            logger.warning(
                "sensitivity_tuning: %s not found in model; skipping.",
                prot_met_id,
            )
            break

        # Find the catalysed rxn consuming the most of this protein.
        # Consumption: S[prot, rxn] < 0 with positive flux -> negative product.
        # We pick the rxn with the most-negative S[prot] * flux.
        target_rxn = None
        target_value = 0.0
        for rxn in prot_met.reactions:
            if rxn.id == best_usage_id:
                continue
            coeff = rxn.metabolites[prot_met]
            flux = float(sol.fluxes.get(rxn.id, 0.0))
            value = coeff * flux
            if value < target_value:
                target_value = value
                target_rxn = rxn

        if target_rxn is None:
            logger.warning(
                "sensitivity_tuning: no catalysed reaction consuming %s; "
                "cannot tune.", prot_met_id,
            )
            break

        # Find the ec entry for this catalysed reaction.
        if target_rxn.id not in model.ec.rxns:
            logger.warning(
                "sensitivity_tuning: rxn %s consumes %s but is not in "
                "model.ec.rxns; cannot tune.",
                target_rxn.id, prot_met_id,
            )
            break
        ec_idx = model.ec.rxns.index(target_rxn.id)

        # Record the pre-tune kcat/source the first time this reaction is
        # tuned, and annotate notes (idempotently).
        if model.ec.source[ec_idx] != _TUNING_SOURCE:
            old_note = model.ec.notes[ec_idx]
            new_note = (
                f"preTuneKcat={_num2str(model.ec.kcat[ec_idx])} | "
                f"source:{model.ec.source[ec_idx]}"
            )
            model.ec.notes[ec_idx] = (
                f"{old_note}; {new_note}" if old_note else new_note
            )

        model.ec.kcat[ec_idx] = float(model.ec.kcat[ec_idx]) * fold_change
        model.ec.source[ec_idx] = _TUNING_SOURCE
        apply_kcat_constraints(model, update_rxns=[target_rxn.id])

        if ec_idx not in tuned_ec_indices:
            tuned_ec_indices.append(ec_idx)
    else:
        logger.warning(
            "sensitivity_tuning: reached max_iterations (%d) without "
            "hitting the target growth rate (%g); stopping.",
            max_iterations, desired_growth_rate,
        )

    return _build_result(model, tuned_ec_indices, input_kcat, input_source)


def _usage_ids_to_ignore(
    usage_rxn_ids: list[str], prot_to_ignore: list[str],
) -> set[str]:
    """Usage reaction ids of ``prot_to_ignore``, matched without regard to
    case; an ID with no usage reaction raises, as MATLAB's ``find`` does."""
    ignore: set[str] = set()
    for prot in prot_to_ignore:
        wanted = f"{USAGE_PREFIX}{prot}".lower()
        hits = [uid for uid in usage_rxn_ids if uid.lower() == wanted]
        if not hits:
            raise ValueError(
                f"prot_to_ignore: no usage reaction for protein {prot!r}."
            )
        ignore.update(hits)
    return ignore


def _light_rxns_of_enzymes(
    model: "EcModel", prot_to_ignore: list[str],
) -> set[str]:
    """Cobra reaction ids that any of the ``prot_to_ignore`` enzymes catalyse
    (gecko-light), matched without regard to case; unknown IDs are ignored."""
    ignore: set[str] = set()
    if not prot_to_ignore:
        return ignore
    mat = model.ec.rxn_enz_mat.tocsc()
    wanted = {p.lower() for p in prot_to_ignore}
    for j, enzyme in enumerate(model.ec.enzymes):
        if enzyme.lower() not in wanted:
            continue
        for row in mat[:, j].nonzero()[0]:
            ignore.add(split_light_rxn_id(model.ec.rxns[int(row)])[1])
    return ignore


def _light_step(
    model: "EcModel", sol, ignore_rxn_ids: set[str],
) -> Optional[list[int]]:
    """The ec rows of the reaction with the most flux through the protein
    pool (gecko-light), or ``None`` when no reaction draws on it."""
    pool = model.metabolites.get_by_id(POOL_ID)
    best_id: Optional[str] = None
    best_value = 0.0
    for rxn in model.reactions:
        if rxn.id in ignore_rxn_ids or pool not in rxn.metabolites:
            continue
        value = rxn.metabolites[pool] * float(sol.fluxes.get(rxn.id, 0.0))
        if value < best_value:
            best_value = value
            best_id = rxn.id
    if best_id is None:
        return None
    rows = [
        i for i, ec_id in enumerate(model.ec.rxns)
        if split_light_rxn_id(ec_id)[1] == best_id
    ]
    return rows or None


def _build_result(
    model: "EcModel",
    ec_indices: list[int],
    input_kcat: np.ndarray,
    input_source: list[str],
) -> TunedKcatsResult:
    """Snapshot input and tuned kcat values of the tuned reactions, listed in
    model order."""
    if not ec_indices:
        return TunedKcatsResult()

    light = bool(model.ec.gecko_light)
    cobra_ids = [
        split_light_rxn_id(model.ec.rxns[i])[1] if light else model.ec.rxns[i]
        for i in ec_indices
    ]
    order = sorted(
        range(len(ec_indices)), key=lambda k: model.reactions.index(cobra_ids[k]),
    )
    ec_indices = [ec_indices[k] for k in order]
    rxns = [cobra_ids[k] for k in order]
    rxn_names = [model.reactions.get_by_id(r).name or r for r in rxns]
    new_kcat = np.array(
        [float(model.ec.kcat[i]) for i in ec_indices], dtype=float,
    )
    old_kcat = np.array([float(input_kcat[i]) for i in ec_indices], dtype=float)
    sources = [input_source[i] for i in ec_indices]

    enzymes: list[str] = []
    rxn_enz_dense = model.ec.rxn_enz_mat.toarray()
    for i in ec_indices:
        enz_idxs = np.where(rxn_enz_dense[i] != 0)[0]
        enzymes.append(";".join(model.ec.enzymes[j] for j in enz_idxs))

    return TunedKcatsResult(
        rxns=rxns,
        rxn_names=rxn_names,
        enzymes=enzymes,
        old_kcat=old_kcat,
        new_kcat=new_kcat,
        source=sources,
    )
