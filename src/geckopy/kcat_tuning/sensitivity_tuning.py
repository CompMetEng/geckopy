"""Bump the most-limiting kcats until the model can grow fast enough.

After ``make_ec_model`` + ``apply_kcat_constraints``, the model
often can't reach the experimental growth rate because one or two
kcat values are unrealistically low (BRENDA's database has many
small or wrong entries). ``sensitivity_tuning`` is the standard
kcat tuning (``cmaes_kcat_tuning`` is the advanced one); it runs the
following loop:

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
src/kcat_tuning/sensitivityTuning.m.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from ..solver_support import optimize_model
from ..ec_model.constants import (
    POOL_EXCHANGE_ID,
    POOL_ID,
    PROT_PREFIX,
    USAGE_PREFIX,
)
from ..ec_model.pipeline.apply_kcat import apply_kcat_constraints
from ..ec_model.pipeline.populate_ec import split_light_rxn_id
from ..gather_kcats.kcat_source_format import format_kcat_source, parse_kcat_source
from .corrections import Correction, corrections as build_corrections

if TYPE_CHECKING:
    import pandas as pd

    from ..databases.flux_data import FluxData
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)


_TUNING_METHOD = "sensitivity"
# Growth is considered stalled when an iteration improves it by no more
# than this (absolute, 1/h). Using a tolerance instead of exact equality
# avoids spinning on sub-numerical-noise increases.
_GROWTH_STALL_TOL = 1e-9
# A measured value smaller than this counts as this size when the stretch of a
# band is weighted, so that a flux measured near zero is not the cheapest thing
# to move by a large amount.
_STRETCH_FLOOR = 0.05


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
    evidence
        With ``evidence=True``, the ``summary`` table of
        :func:`~geckopy.gather_kcats.kcat_evidence` for the tuned reactions
        (``None`` otherwise).
    status
        Why the run stopped: ``reached`` (the target is met),
        ``infeasible_in_gem`` (the measured condition is infeasible without
        any enzyme limit, which is the model without enzyme constraints, so no
        kcat can meet it and the condition should not be used),
        ``unfixable`` (feasible without enzyme limits and infeasible with the
        protein pool lifted, so a limit of single enzymes stops it),
        ``stalled`` (the next kcat does not lower the protein needed),
        ``infeasible`` or ``no_candidate`` (the LP has no optimal solution,
        or no enzyme is left to raise) and ``max_iterations``.
    conflicts
        With status ``infeasible_in_gem``, the table of
        :func:`diagnose_condition`: the measured values that have to move for
        the condition to become feasible (``None`` otherwise).
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
    evidence: Optional["pd.DataFrame"] = None
    status: str = ""
    conflicts: Optional["pd.DataFrame"] = None

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
        ``rxn_names`` already contains resolved names, so there is no
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
    evidence: bool = False,
    flux_data: Optional["FluxData"] = None,
    condition: Union[str, int] = 0,
    flux_tolerance: float = 0.1,
    flux_abs_tolerance: float = 1e-3,
    flux_tolerances: Optional[Mapping[str, float]] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
) -> TunedKcatsResult:
    """Iteratively bump the most-limiting kcat by ``fold_change`` until
    the model can reach ``desired_growth_rate``.

    This is the standard kcat tuning: it needs one maximum growth rate.
    :func:`~geckopy.kcat_tuning.evotune.cmaes_kcat_tuning` is the advanced
    tuning, which fits kcats to maximum growth rates and exchange fluxes,
    measured in one or more conditions.

    Each iteration:

    1. Solves the LP at the current kcat values.
    2. If growth didn't increase since the last iteration, warns and
       breaks (likely a substrate / pool constraint, not a kcat one).
    3. If growth reached the target, returns.
    4. Otherwise selects the ``usage_prot_*`` reaction with the highest
       flux (= most-produced enzyme), excluding any whose UniProt ID
       is in ``prot_to_ignore``.
    5. Finds the catalysed reaction consuming the most of that
       enzyme's pseudometabolite and multiplies that reaction's
       ``ec.kcat`` by ``fold_change``. A tuned kcat has no upper limit.
    6. Extends the tuning status of ``ec.source`` with ``sensitivity``
       (the origin and detail are kept; the kcat before the first tuning
       is recorded as ``prior`` unless the source already has one).
    7. Re-applies kcat constraints for that single reaction.

    On a gecko-light model each iteration instead selects the reaction with
    the most flux through the protein pool, multiplies the kcat of all its
    isozyme rows by ``fold_change`` and re-applies the constraint of that
    reaction; the tuning status of ``ec.source`` is extended for each of those
    rows as above.

    Ported from GECKO MATLAB:
    src/kcat_tuning/sensitivityTuning.m.

    Parameters
    ----------
    model
        EcModel with the protein pool / usage rxn machinery
        installed and a populated ``ec.kcat``. Mutated in place.
    desired_growth_rate
        Target growth rate. Defaults to
        ``model.adapter.params.gr_exp``; raises ``ValueError`` when neither is set.
    fold_change
        Per-iteration kcat multiplier.
    prot_to_ignore
        UniProt IDs whose usage rxns are excluded from being selected
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
    evidence
        Add ``TunedKcatsResult.evidence``, the kcats that every source file of
        the project gives for the tuned reactions next to the tuned kcat (see
        :func:`~geckopy.gather_kcats.kcat_evidence`). A failure to read the
        source files is logged as a warning and leaves ``evidence`` empty.

    flux_data, condition
        Measured growth rate and exchange fluxes of one condition (see
        :func:`~geckopy.databases.flux_data.load_flux_data`), given as the
        condition name or its row. With ``flux_data`` the target is the
        whole condition: growth at the measured rate, and every measured
        exchange within ``flux_tolerance`` of its value. The model's own
        medium is kept apart from the measured exchanges, which take the
        bounds of the condition (an exchange of ``-1000`` or ``1000`` stays
        unconstrained in that direction, and ``NaN`` is not measured).
        ``desired_growth_rate`` replaces the measured growth rate. Each
        iteration checks the condition; when it is infeasible the protein
        pool is lifted as a limit, the protein drawn from it is minimised,
        and the kcat of the enzyme that draws the most is raised. A
        condition that is infeasible without any enzyme limit, which is the
        model without enzyme constraints, is not something a kcat can fix (a
        measurement that conflicts with the model): the run stops before any
        change, with status ``infeasible_in_gem``, and the condition is better
        left out of the tuning (give it ``weight`` 0, see
        :func:`infeasible_conditions`). A condition that is feasible without
        enzyme limits but not with the pool lifted is stopped by a limit of
        single enzymes, status ``unfixable``. Without ``flux_data`` the
        behaviour is as described above.
    make_anaerobic, change_protein_biomass
        Organism-specific setup of a condition, as in
        :func:`~geckopy.kcat_tuning.evotune.cmaes_kcat_tuning`: the first runs
        in a condition whose oxygen exchange is measured as exactly 0, the
        second with the total protein content ``Ptot`` of a condition that has
        one. A condition that needs one of them and gets none is simulated
        without it. The changes apply only while the condition is checked.
    flux_tolerance, flux_abs_tolerance, flux_tolerances
        A measured flux ``v`` is met within ``[v(1 - t), v(1 + t)]`` widened
        by ``flux_abs_tolerance`` (mmol/gDW/h) on both sides, with ``t`` the
        ``flux_tolerance`` or the entry of ``flux_tolerances`` for the
        reaction id. Fluxes near zero need the absolute part.

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

    cond_idx = None
    if flux_data is not None:
        cond_idx = _condition_index(flux_data, condition)
        if desired_growth_rate is None:
            desired_growth_rate = float(flux_data.gr_rate[cond_idx])

    if desired_growth_rate is None:
        from ..adapter import resolve_gr_exp
        desired_growth_rate = resolve_gr_exp(
            model,
            purpose="sensitivity_tuning needs `desired_growth_rate` "
            "(pass it explicitly, or set gr_exp in the adapter)",
        )

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

    if flux_data is not None:
        tuned_ec_indices, status, conflicts = _tune_to_condition(
            model, flux_data, cond_idx, float(desired_growth_rate),
            flux_tolerance, flux_abs_tolerance, flux_tolerances or {},
            fold_change, max_iterations, usage_rxn_ids, ignore_usage_ids,
            ignore_rxn_ids, light, bio_rxn_id, verbose,
            make_anaerobic, change_protein_biomass,
        )
        return _finish(model, tuned_ec_indices, input_kcat, input_source,
                       evidence, status, conflicts)

    last_growth = float("nan")
    status = "max_iterations"
    # Indices into model.ec.rxns; on a light model the first isozyme row of
    # each tuned reaction.
    tuned_ec_indices: list[int] = []
    iteration = 1

    while iteration <= max_iterations:
        sol = optimize_model(model)
        if sol.status != "optimal":
            logger.warning(
                "sensitivity_tuning: the model has no optimal solution with the "
                "increased kcats (status %s); tuning stops.", sol.status,
            )
            status = "infeasible"
            break
        growth = float(sol.objective_value)

        if not np.isnan(last_growth) and growth <= last_growth + _GROWTH_STALL_TOL:
            logger.warning(
                "sensitivity_tuning: growth did not increase from %g; "
                "uptake or pool constraints may be limiting.", growth,
            )
            status = "stalled"
            break
        last_growth = growth
        if verbose:
            logger.info(
                "sensitivity_tuning: iteration %d, growth = %g",
                iteration, growth,
            )
        if growth >= desired_growth_rate:
            status = "reached"
            break

        iteration += 1

        if light:
            step = _light_step(model, sol, ignore_rxn_ids)
            if step is None:
                logger.warning(
                    "sensitivity_tuning: no reaction draws on the protein "
                    "pool; cannot continue tuning."
                )
                status = "no_candidate"
                break
            ec_rows = step
            for i in ec_rows:
                _mark_tuned(model, i)
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
            status = "no_candidate"
            break

        ec_idx = _bump_usage_enzyme(model, best_usage_id, sol, fold_change)
        if ec_idx is None:
            status = "no_candidate"
            break
        if ec_idx not in tuned_ec_indices:
            tuned_ec_indices.append(ec_idx)
    else:
        logger.warning(
            "sensitivity_tuning: reached max_iterations (%d) without "
            "hitting the target growth rate (%g); stopping.",
            max_iterations, desired_growth_rate,
        )

    return _finish(model, tuned_ec_indices, input_kcat, input_source, evidence, status, None)


def _finish(
    model: "EcModel", tuned_ec_indices: list[int], input_kcat: np.ndarray,
    input_source: list[str], evidence: bool, status: str,
    conflicts: Optional["pd.DataFrame"],
) -> TunedKcatsResult:
    result = _build_result(model, tuned_ec_indices, input_kcat, input_source)
    result.status = status
    result.conflicts = conflicts
    if evidence:
        from ..gather_kcats.evidence import kcat_evidence

        try:
            result.evidence = kcat_evidence(model, tuned=result).summary
        except Exception as err:  # the tuning is done; the evidence is extra
            logger.warning("The evidence of the tuned kcats could not be read: %s", err)
    return result


def _condition_index(flux_data: "FluxData", condition: Union[str, int]) -> int:
    if isinstance(condition, str):
        if condition not in flux_data.conds:
            raise ValueError(
                f"Condition {condition!r} not found in flux_data.conds "
                f"({flux_data.conds})"
            )
        return flux_data.conds.index(condition)
    if not 0 <= condition < len(flux_data.conds):
        raise IndexError(
            f"condition index {condition} out of range "
            f"(have {len(flux_data.conds)} condition(s))"
        )
    return int(condition)


def _condition_bounds(
    flux_data: "FluxData", cond_idx: int, tol: float, abs_tol: float,
    per_flux: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Bounds that hold every measured exchange of a condition within its band."""
    bounds: dict[str, tuple[float, float]] = {}
    for rxn_id, value in zip(flux_data.exch_rxn_ids, flux_data.exch_fluxes[cond_idx]):
        if np.isnan(value):
            continue
        if abs(value) == 1000.0:
            bounds[rxn_id] = (-1000.0, 0.0) if value < 0 else (0.0, 1000.0)
            continue
        t = float(per_flux.get(rxn_id, tol))
        lo, hi = sorted((value * (1 - t), value * (1 + t)))
        bounds[rxn_id] = (lo - abs_tol, hi + abs_tol)
    return bounds


def _setup_condition(
    model: "EcModel", flux_data: "FluxData", cond_idx: int,
    make_anaerobic: Optional[Callable[["EcModel"], None]],
    change_protein_biomass: Optional[Callable[["EcModel", float], None]],
) -> None:
    """Apply the anaerobic switch and the protein content of a condition. Call
    inside ``with model:``."""
    if make_anaerobic is not None and "oxygen" in flux_data.exch_mets:
        o2 = flux_data.exch_mets.index("oxygen")
        if flux_data.exch_fluxes[cond_idx, o2] == 0.0:
            make_anaerobic(model)
    ptot = float(flux_data.p_tot[cond_idx])
    if change_protein_biomass is not None and not (np.isnan(ptot) or ptot == 0.0):
        change_protein_biomass(model, ptot)


def _feasible_without_enzyme_limits(
    model: "EcModel", bounds: Mapping[str, tuple[float, float]], target_growth: float,
    bio_rxn_id: str, usage_rxn_ids: Sequence[str], setup: Optional[Callable] = None,
) -> bool:
    """Whether the condition is feasible with the protein pool and the usage of
    every enzyme unlimited, which is the model without enzyme constraints."""
    with model:
        if setup is not None:
            setup()
        for rxn_id, (lo, hi) in bounds.items():
            model.reactions.get_by_id(rxn_id).bounds = (lo, hi)
        model.reactions.get_by_id(bio_rxn_id).lower_bound = target_growth
        model.reactions.get_by_id(POOL_EXCHANGE_ID).upper_bound = 1000.0
        for uid in usage_rxn_ids:
            model.reactions.get_by_id(uid).bounds = (0.0, 1000.0)
        model.objective = bio_rxn_id
        model.slim_optimize(error_value=float("nan"))
        return model.solver.status == "optimal"


def infeasible_conditions(
    model: "EcModel",
    flux_data: "FluxData",
    *,
    desired_growth_rate: Optional[float] = None,
    flux_tolerance: float = 0.1,
    flux_abs_tolerance: float = 1e-3,
    flux_tolerances: Optional[Mapping[str, float]] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
) -> list[str]:
    """Names of the conditions that are infeasible in the model without enzyme limits.

    The check is the one :func:`sensitivity_tuning` makes first, for every
    condition of ``flux_data``: growth at the measured rate (or
    ``desired_growth_rate``) with every measured exchange within its band. No
    kcat can fix such a condition. Give these conditions ``weight`` 0 in the
    flux data, so that no tuning is asked to fit them. ``model`` is not
    changed. ``make_anaerobic`` and ``change_protein_biomass`` are the
    condition setup of :func:`sensitivity_tuning`.
    """
    bio_rxn_id = bio_rxn
    if bio_rxn_id is None and model.adapter is not None:
        bio_rxn_id = model.adapter.params.bio_rxn
    if not bio_rxn_id:
        raise ValueError("The biomass reaction (bio_rxn) is needed.")
    usage_rxn_ids = [r.id for r in model.reactions if r.id.startswith(USAGE_PREFIX)]
    infeasible = []
    for i, name in enumerate(flux_data.conds):
        bounds = _condition_bounds(
            flux_data, i, flux_tolerance, flux_abs_tolerance, flux_tolerances or {})
        target = float(flux_data.gr_rate[i]) if desired_growth_rate is None \
            else float(desired_growth_rate)
        setup = lambda i=i: _setup_condition(  # noqa: E731
            model, flux_data, i, make_anaerobic, change_protein_biomass)
        if not _feasible_without_enzyme_limits(
                model, bounds, target, bio_rxn_id, usage_rxn_ids, setup):
            infeasible.append(name)
    return infeasible


def diagnose_condition(
    model: "EcModel",
    flux_data: "FluxData",
    condition: Union[str, int] = 0,
    *,
    desired_growth_rate: Optional[float] = None,
    flux_tolerance: float = 0.1,
    flux_abs_tolerance: float = 1e-3,
    flux_tolerances: Optional[Mapping[str, float]] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
) -> "pd.DataFrame":
    """Which measured values of a condition conflict with the model.

    In the model without enzyme limits (the protein pool and the usage of
    every enzyme unlimited) the bands of the measured exchanges and the
    growth rate are allowed to stretch, at a cost of the stretch relative to
    the measured value, and the total cost is minimised. The measurements
    that have to move for the condition to become feasible are the ones that
    conflict with the model, with its constraints or with each other.

    One row per measured exchange and one for growth: ``reaction``,
    ``measured``, ``band_low`` and ``band_high``, ``achieved`` (the flux of the
    least stretched solution), ``stretch`` (the distance from the achieved
    flux to the band, 0 inside it) and ``relative_stretch`` (``stretch``
    divided by the size of the measured value, where a value below 0.05
    counts as 0.05), sorted with the largest stretch first. A minimal total
    stretch is not unique in general, so the table names one way to make the
    condition feasible, not every way. ``model`` is not changed.
    """
    cond_idx = _condition_index(flux_data, condition)
    bio_rxn_id = bio_rxn
    if bio_rxn_id is None and model.adapter is not None:
        bio_rxn_id = model.adapter.params.bio_rxn
    if not bio_rxn_id:
        raise ValueError("The biomass reaction (bio_rxn) is needed.")
    target = float(flux_data.gr_rate[cond_idx]) if desired_growth_rate is None \
        else float(desired_growth_rate)
    bounds = _condition_bounds(
        flux_data, cond_idx, flux_tolerance, flux_abs_tolerance, flux_tolerances or {})
    measured = {
        r: float(v) for r, v in zip(flux_data.exch_rxn_ids, flux_data.exch_fluxes[cond_idx])
        if not np.isnan(v) and abs(v) != 1000.0
    }
    usage_rxn_ids = [r.id for r in model.reactions if r.id.startswith(USAGE_PREFIX)]
    rows = []
    with model:
        _setup_condition(model, flux_data, cond_idx, make_anaerobic, change_protein_biomass)
        if POOL_EXCHANGE_ID in model.reactions:
            model.reactions.get_by_id(POOL_EXCHANGE_ID).upper_bound = 1000.0
        for uid in usage_rxn_ids:
            model.reactions.get_by_id(uid).bounds = (0.0, 1000.0)
        prob = model.problem
        slacks, cost_terms, new_vars, new_cons = {}, [], [], []
        bands = {r: bounds[r] for r in measured}
        bands[bio_rxn_id] = (target, float("inf"))
        sizes = {r: max(abs(v), _STRETCH_FLOOR) for r, v in measured.items()}
        sizes[bio_rxn_id] = max(abs(target), _STRETCH_FLOOR)
        for rxn_id, (lo, hi) in bands.items():
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.bounds = (-1000.0, 1000.0) if rxn_id != bio_rxn_id else (0.0, 1000.0)
            s_lo = prob.Variable(f"stretch_lo_{rxn_id}", lb=0.0)
            s_hi = prob.Variable(f"stretch_hi_{rxn_id}", lb=0.0)
            new_vars += [s_lo, s_hi]
            new_cons.append(prob.Constraint(rxn.flux_expression + s_lo, lb=lo, name=f"band_lo_{rxn_id}"))
            if np.isfinite(hi):
                new_cons.append(prob.Constraint(rxn.flux_expression - s_hi, ub=hi, name=f"band_hi_{rxn_id}"))
            cost_terms += [(s_lo / sizes[rxn_id]), (s_hi / sizes[rxn_id])]
            slacks[rxn_id] = (s_lo, s_hi)
        model.add_cons_vars(new_vars + new_cons)
        model.objective = prob.Objective(sum(cost_terms), direction="min")
        sol = optimize_model(model)
        if sol.status != "optimal":
            raise ValueError(
                "The condition is infeasible even with every band stretched "
                f"(status {sol.status!r}); the model itself blocks it."
            )
        for rxn_id, (lo, hi) in bands.items():
            achieved = float(sol.fluxes[rxn_id])
            stretch = max(lo - achieved, 0.0, achieved - hi if np.isfinite(hi) else 0.0)
            rows.append({
                "reaction": rxn_id,
                "measured": target if rxn_id == bio_rxn_id else measured[rxn_id],
                "band_low": lo, "band_high": hi, "achieved": achieved,
                "stretch": stretch, "relative_stretch": stretch / sizes[rxn_id],
            })
    import pandas as pd

    table = pd.DataFrame(rows)
    return table.sort_values("relative_stretch", ascending=False).reset_index(drop=True)


def _tune_to_condition(
    model: "EcModel", flux_data: "FluxData", cond_idx: int, target_growth: float,
    tol: float, abs_tol: float, per_flux: Mapping[str, float], fold_change: float,
    max_iterations: int, usage_rxn_ids: list[str], ignore_usage_ids: set[str],
    ignore_rxn_ids: set[str], light: bool, bio_rxn_id: Optional[str], verbose: bool,
    make_anaerobic: Optional[Callable] = None,
    change_protein_biomass: Optional[Callable] = None,
) -> tuple[list[int], str, Optional["pd.DataFrame"]]:
    """Raise kcats until the model meets growth and the measured fluxes of a condition."""
    if not bio_rxn_id:
        raise ValueError("A condition needs the biomass reaction (bio_rxn).")
    if POOL_EXCHANGE_ID not in model.reactions:
        raise ValueError(
            f"The model has no {POOL_EXCHANGE_ID} reaction, which the tuning to "
            "a condition needs."
        )
    bounds = _condition_bounds(flux_data, cond_idx, tol, abs_tol, per_flux)
    missing = [r for r in bounds if r not in model.reactions]
    if missing:
        raise ValueError(
            f"{len(missing)} exchange reaction ID(s) of the condition are not in "
            f"the model (examples: {missing[:5]})"
        )
    pool_met = model.metabolites.get_by_id(POOL_ID)
    pool_exchange = model.reactions.get_by_id(POOL_EXCHANGE_ID)
    bio_rxn = model.reactions.get_by_id(bio_rxn_id)
    tuned: list[int] = []
    last_pool = float("nan")

    def _condition():
        _setup_condition(model, flux_data, cond_idx, make_anaerobic, change_protein_biomass)
        for rxn_id, (lo, hi) in bounds.items():
            model.reactions.get_by_id(rxn_id).bounds = (lo, hi)
        bio_rxn.lower_bound = target_growth

    if not _feasible_without_enzyme_limits(
            model, bounds, target_growth, bio_rxn_id, usage_rxn_ids,
            lambda: _setup_condition(
                model, flux_data, cond_idx, make_anaerobic, change_protein_biomass)):
        logger.warning(
            "sensitivity_tuning: the condition is infeasible without any enzyme "
            "limit, which is the model without enzyme constraints, so no kcat "
            "can meet it; check the measured fluxes against the model."
        )
        conflicts = diagnose_condition(
            model, flux_data, cond_idx, desired_growth_rate=target_growth,
            flux_tolerance=tol, flux_abs_tolerance=abs_tol, flux_tolerances=per_flux,
            bio_rxn=bio_rxn_id, make_anaerobic=make_anaerobic,
            change_protein_biomass=change_protein_biomass)
        return tuned, "infeasible_in_gem", conflicts

    for iteration in range(1, max_iterations + 1):
        with model:
            _condition()
            model.objective = bio_rxn_id
            model.slim_optimize(error_value=float("nan"))
            feasible = model.solver.status == "optimal"
        if feasible:
            if verbose:
                logger.info("sensitivity_tuning: the condition is met after %d "
                            "iteration(s)", iteration)
            return tuned, "reached", None

        with model:
            _condition()
            pool_exchange.upper_bound = 1000.0
            model.objective = {pool_exchange: 1.0}
            model.objective_direction = "min"
            sol = optimize_model(model)
            optimal = sol.status == "optimal"
        if not optimal:
            logger.warning(
                "sensitivity_tuning: the condition is feasible without enzyme limits "
                "but infeasible with the protein pool lifted (status %s), so a "
                "limit of single enzymes stops it.", sol.status,
            )
            return tuned, "unfixable", None
        pool_used = float(sol.objective_value)
        if not np.isnan(last_pool) and pool_used >= last_pool - 1e-9:
            logger.warning(
                "sensitivity_tuning: the protein needed for the condition did not "
                "fall from %g; tuning stops.", pool_used,
            )
            return tuned, "stalled", None
        last_pool = pool_used
        if verbose:
            logger.info("sensitivity_tuning: iteration %d, the condition needs %g "
                        "of protein", iteration, pool_used)

        if light:
            rows = _light_step(model, sol, ignore_rxn_ids)
            if rows is None:
                logger.warning("sensitivity_tuning: no reaction draws on the "
                               "protein pool; cannot continue tuning.")
                return tuned, "no_candidate", None
            for i in rows:
                _mark_tuned(model, i)
                model.ec.kcat[i] = float(model.ec.kcat[i]) * fold_change
            apply_kcat_constraints(model, update_rxns=[model.ec.rxns[rows[0]]])
            if rows[0] not in tuned:
                tuned.append(rows[0])
            continue

        best_id, best_draw = None, 0.0
        for uid in usage_rxn_ids:
            if uid in ignore_usage_ids:
                continue
            rxn = model.reactions.get_by_id(uid)
            draw = float(sol.fluxes.get(uid, 0.0)) * abs(rxn.metabolites.get(pool_met, 1.0))
            if draw > best_draw:
                best_id, best_draw = uid, draw
        if best_id is None:
            logger.warning("sensitivity_tuning: no usage rxn carries flux; cannot "
                           "continue tuning.")
            return tuned, "no_candidate", None
        ec_idx = _bump_usage_enzyme(model, best_id, sol, fold_change)
        if ec_idx is None:
            return tuned, "no_candidate", None
        if ec_idx not in tuned:
            tuned.append(ec_idx)
    logger.warning(
        "sensitivity_tuning: reached max_iterations (%d) without meeting the "
        "condition; stopping.", max_iterations,
    )
    return tuned, "max_iterations", None


def _bump_usage_enzyme(
    model: "EcModel", usage_id: str, sol, fold_change: float,
) -> Optional[int]:
    """Multiply the kcat of the reaction that draws most on the enzyme of
    ``usage_id`` by ``fold_change``; the index into ``ec.rxns``, or ``None``
    (with a warning) when there is no such reaction."""
    best_usage_id = usage_id
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
        return None

    # Find the catalysed rxn consuming the most of this protein.
    # Consumption: S[prot, rxn] < 0 with positive flux -> negative product.
    # The rxn with the most-negative S[prot] * flux is selected.
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
        return None

    # Find the ec entry for this catalysed reaction.
    if target_rxn.id not in model.ec.rxns:
        logger.warning(
            "sensitivity_tuning: rxn %s consumes %s but is not in "
            "model.ec.rxns; cannot tune.",
            target_rxn.id, prot_met_id,
        )
        return None
    ec_idx = model.ec.rxns.index(target_rxn.id)

    kcat_before = float(model.ec.kcat[ec_idx])
    _mark_tuned(model, ec_idx)
    model.ec.kcat[ec_idx] = kcat_before * fold_change
    apply_kcat_constraints(model, update_rxns=[target_rxn.id])
    return ec_idx


def _mark_tuned(model: "EcModel", ec_idx: int) -> None:
    """Add the tuning method to the source of one ec row.

    The kcat before the first tuning is the prior. Called before the kcat
    changes.
    """
    previous = parse_kcat_source(model.ec.source[ec_idx])
    model.ec.source[ec_idx] = format_kcat_source(
        previous.origin, previous.detail,
        dict.fromkeys((*previous.tuned, _TUNING_METHOD)),
        previous.prior if previous.prior is not None else float(model.ec.kcat[ec_idx]),
    )


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
