"""Loosen proteomics constraints until the model can grow.

After ``constrain_enz_concs`` pins each measured enzyme to its
proteomics value, the model often can't reach the experimental
growth rate any more — the measurements are noisy, and a few
enzymes end up too tightly capped to feed the network.

``flexibilize_enz_concs`` fixes that automatically. It computes a
control coefficient per enzyme (how much each enzyme's bound is
limiting growth), picks the worst one, multiplies its upper bound
by ``fold_change``, re-solves, and repeats until growth reaches
the target. The result records which enzymes were loosened and by
how much, so a curator can review.

A related function is ``relax_proteomics_greedy`` in the same
package: same goal, different algorithm. This one ranks by control
coefficient, relaxes gradually, and runs a tighten-back pass, so it
yields the *minimal* (most proteomics-faithful) relaxation but is
slower. ``relax_proteomics_greedy`` ranks by shadow price and fully
unconstrains each picked enzyme — faster, but looser. See
``docs/relaxation_methods.md`` for the full comparison.

Ported from GECKO MATLAB:
src/geckomat/limit_proteins/flexibilizeEnzConcs.m.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..ec_model.constants import (
    POOL_EXCHANGE_ID,
    USAGE_PREFIX,
)
from .get_conc_control_coeffs import get_conc_control_coeffs
from .level_complex_concs import ComplexFamilies, complex_families

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)

_POOL_TARGET_NAME = "prot_pool"
_GROWTH_DELTA_THRESHOLD = 1e-3
# Backstop on total iterations when iter_per_enzyme is infinite (the
# MATLAB-compat default for iter_per_enzyme=0). Prevents an unreachable
# exp_growth from spinning forever when control coefficients keep
# flagging different enzymes.
_ABSOLUTE_ITER_CAP_PER_ENZYME = 1000
_CONTROL_COEFF_LIMIT = 0.75
# Half-width (as a percentage of exp_growth, RAVEN setParam's 'var' units)
# of the soft band the post-loop refinement solve allows bio_rxn to land
# in, rather than pinning it to exp_growth exactly. Matches MATLAB's
# flexibilizeEnzConcs.m, which calls
# setParam(model, 'var', bioRxn, expGrowth, 0.5): setParam's 'var' case
# sets lb/ub to params*(1 -/+ var/200), so var=0.5 is a +/-0.25% band
# (raven-gecko-parity#71).
_REFINEMENT_GROWTH_BAND_VAR = 0.5


@dataclass
class FlexEnzResult:
    """Information about which enzyme constraints were flexibilized.

    All arrays are sorted by ``ratio_incr`` descending. May contain a
    final entry with ``uniprot_ids[-1] == "prot_pool"`` if the protein
    pool exchange itself was relaxed.

    Attributes
    ----------
    uniprot_ids
        UniProt IDs of the flexibilized enzymes (plus ``"prot_pool"``
        as a final entry when the protein pool itself was relaxed).
    old_concs
        Upper bound of each enzyme's usage reaction (or of
        ``prot_pool_exchange`` for the ``"prot_pool"`` entry) before
        relaxation.
    flex_concs
        Upper bound after relaxation.
    ratio_incr
        ``flex_concs / old_concs``; ``inf`` where ``old_concs`` is 0.
    frequence
        Number of loop iterations in which each enzyme was picked as
        the most limiting one.
    """

    uniprot_ids: list[str] = field(default_factory=list)
    old_concs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    flex_concs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ratio_incr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    frequence: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def flexibilize_enz_concs(
    model: "EcModel",
    *,
    exp_growth: Optional[float] = None,
    fold_change: float = 2.0,
    iter_per_enzyme: int = 5,
    bio_rxn: Optional[str] = None,
    verbose: bool = True,
    per_complex: bool = False,
) -> FlexEnzResult:
    """Iteratively relax enzyme concentration constraints until the
    model reaches ``exp_growth``.

    Each iteration uses ``get_conc_control_coeffs`` to find the most
    limiting enzyme, then multiplies its ``usage_prot_<enzyme>``
    upper bound by ``(1 + fold_change * iteration_count)``. If no
    enzyme is limiting (all control coefficients zero), the function
    tries relaxing ``prot_pool_exchange``.

    After the loop, a refinement pass solves ``min prot_pool_exchange``
    with ``bio_rxn`` constrained to a soft +/-0.25% band around
    ``exp_growth`` (matching MATLAB's ``flexibilizeEnzConcs.m``, which
    uses RAVEN's ``setParam('var', bioRxn, expGrowth, 0.5)`` for the same
    solve) to find the minimal usage that supports growth. Because
    minimal protein-pool usage is non-decreasing in growth rate near the
    optimum, the minimiser typically settles at the band's low edge --
    growth ends up about 0.25% below ``exp_growth``, not at it exactly.
    For each flexibilized enzyme, if the actual usage at this minimum is
    less than its original concentration, the original constraint is
    restored (i.e. that enzyme is dropped from the result). This avoids
    over-relaxing enzymes that the prior iterative pass increased more
    than necessary.

    Ported from GECKO MATLAB:
    src/geckomat/limit_proteins/flexibilizeEnzConcs.m.

    Parameters
    ----------
    model
        EcModel with `prot_pool_exchange` and ``usage_prot_*``
        reactions installed and ``model.ec.concs`` populated for at
        least some enzymes (typically by ``fill_enz_concs`` and
        ``constrain_enz_concs``). Mutated in place; ``ec.concs``
        itself is left unchanged.
    exp_growth
        Target growth rate. Defaults to
        ``model.adapter.params.gr_exp``.
    fold_change
        Controls how much the limiting enzyme's usage upper bound
        grows each time it is picked: on its ``k``-th pick, the
        bound is set to ``original_conc * (1 + fold_change * k)``.
    iter_per_enzyme
        Maximum number of times the same enzyme may be picked before
        warning and breaking. ``0`` means no limit.
    bio_rxn
        Biomass reaction id. Defaults to ``params.bio_rxn`` from the
        adapter; pass it (with ``exp_growth``) to run without an adapter.
    verbose
        Whether per-iteration progress is logged at INFO.
    per_complex
        Treat every complex (see ``complex_families``) as one enzyme:
        rank enzymes and complexes by the growth gained per fold of
        raising all their caps together (a levelled complex binds on all
        subunits at once, so no single subunit shows a gain), raise every
        measured subunit of the picked complex by the same factor, count
        iterations per complex, and keep one factor per complex in the
        refinement pass, so that concentrations levelled by
        ``level_complex_concs`` stay levelled. Off by default.

    Returns
    -------
    FlexEnzResult

    Raises
    ------
    ValueError
        If no enzyme has a measured concentration (nothing to
        flexibilize), or if ``model.adapter`` is None and
        ``exp_growth`` is also None.
    """
    from ..adapter import resolve_param
    if exp_growth is None:
        exp_growth = resolve_param(
            model, None, "gr_exp",
            purpose="flexibilize_enz_concs needs `exp_growth` (pass it "
            "explicitly, or rely on params.gr_exp from the adapter)",
        )
        if exp_growth is None:
            raise ValueError(
                "exp_growth not provided and params.gr_exp is unset."
            )
    exp_growth = float(exp_growth)

    if iter_per_enzyme == 0:
        iter_per_enzyme = math.inf

    bio_rxn_id = resolve_param(
        model, bio_rxn, "bio_rxn",
        purpose="flexibilize_enz_concs needs the biomass reaction id",
    )
    bio_rxn = model.reactions.get_by_id(bio_rxn_id)
    if bio_rxn.upper_bound < exp_growth:
        logger.info(
            "flexibilize_enz_concs: bio_rxn upper_bound (%g) was below "
            "exp_growth (%g); raising it.",
            bio_rxn.upper_bound, exp_growth,
        )
        bio_rxn.upper_bound = exp_growth

    measured_idx = np.where(~np.isnan(model.ec.concs))[0]
    if len(measured_idx) == 0:
        raise ValueError(
            "No enzyme concentrations are measured (model.ec.concs is "
            "all-NaN). Run fill_enz_concs and constrain_enz_concs first."
        )
    proteins = [model.ec.enzymes[i] for i in measured_idx]
    n_proteins = len(proteins)
    frequence = np.zeros(n_proteins, dtype=int)
    touched = np.zeros(n_proteins, dtype=bool)
    position = {p: i for i, p in enumerate(proteins)}
    families: Optional[ComplexFamilies] = complex_families(model) if per_complex else None
    # Indices into ``proteins`` of each unit that is relaxed together: a
    # complex's measured, non-shared subunits, or a single enzyme.
    units: list[list[int]] = []
    unit_of = np.zeros(n_proteins, dtype=int)
    assigned: set[int] = set()
    for i, protein in enumerate(proteins):
        if i in assigned:
            continue
        members = [position[m] for m in families.members(protein) if m in position] if families else [i]
        for j in members:
            unit_of[j] = len(units)
        assigned.update(members)
        units.append(members)
    unit_frequence = np.zeros(len(units), dtype=int)

    initial_sol = model.optimize()
    if (
        initial_sol.objective_value is None
        or np.isnan(initial_sol.objective_value)
    ):
        raise ValueError(
            "FBA of the input model is infeasible, so there is no growth to "
            "flexibilize towards. Check the enzyme-concentration and exchange "
            "constraints before calling flexibilize_enz_concs."
        )
    pred_growth = float(initial_sol.objective_value)

    flex_break = False
    pool_old: Optional[float] = None
    pool_new: Optional[float] = None

    cap_per_enzyme = (
        iter_per_enzyme if math.isfinite(iter_per_enzyme)
        else _ABSOLUTE_ITER_CAP_PER_ENZYME
    )
    max_total_iter = n_proteins * cap_per_enzyme + n_proteins
    iter_count = 0

    while pred_growth < exp_growth:
        iter_count += 1
        if iter_count > max_total_iter:
            logger.warning(
                "flexibilize_enz_concs: hit the global iteration cap (%d) "
                "without reaching exp_growth (%g); stopping. The target may "
                "be unreachable even with relaxed enzyme bounds.",
                int(max_total_iter), exp_growth,
            )
            flex_break = True
            break
        if families is not None:
            scores = _unit_scores(model, proteins, units, fold_change, _CONTROL_COEFF_LIMIT)
        else:
            _, scores = get_conc_control_coeffs(
                model,
                proteins=proteins,
                fold_change=fold_change,
                limit=_CONTROL_COEFF_LIMIT,
            )

        if not (scores > 0).any():
            pool_old, pool_new, pool_pred_growth = _try_relax_pool(
                model, bio_rxn, exp_growth, pred_growth, verbose,
            )
            if pool_pred_growth is not None:
                pred_growth = pool_pred_growth
            # No enzyme limits growth any more: the highest growth reached is the
            # target of the final pass and of the reported model, whether or not
            # exp_growth was reached.
            exp_growth = pred_growth
            break

        if families is not None:
            unit = int(np.argmax(scores))
        else:
            unit = int(unit_of[int(np.argmax(scores))])
        members = units[unit]
        max_idx = members[0]
        frequence[max_idx] += 1
        unit_frequence[unit] += 1
        if unit_frequence[unit] > iter_per_enzyme:
            logger.warning(
                "flexibilize_enz_concs: iter limit reached for %s %r; "
                "consider revising its kcat.",
                "complex" if len(members) > 1 else "protein",
                [proteins[j] for j in members] if len(members) > 1 else proteins[max_idx],
            )
            flex_break = True
            break

        increase = fold_change * unit_frequence[unit]
        for i in members:
            touched[i] = True
            usage_rxn = model.reactions.get_by_id(f"{USAGE_PREFIX}{proteins[i]}")
            usage_rxn.upper_bound = max(
                usage_rxn.upper_bound,
                float(model.ec.concs[measured_idx[i]]) * (1.0 + increase),
            )

        new_sol = model.optimize()
        pred_growth = float(new_sol.objective_value or 0.0)
        if verbose:
            logger.info(
                "flexibilize_enz_concs: protein %r relaxed; growth = %g",
                proteins[max_idx], pred_growth,
            )

    return _build_result(
        model, proteins, measured_idx, frequence,
        flex_break, exp_growth, pool_old, pool_new, bio_rxn_id,
        touched=touched, families=families,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _try_relax_pool(
    model: "EcModel",
    bio_rxn,
    exp_growth: float,
    pred_growth: float,
    verbose: bool,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Try relaxing prot_pool_exchange. Returns (old_pool, new_pool, growth)
    if a relaxation was applied; (None, None, None) otherwise."""
    pool_rxn = model.reactions.get_by_id(POOL_EXCHANGE_ID)
    old_pool = pool_rxn.upper_bound

    with model:
        pool_rxn.upper_bound = 1000.0
        bio_rxn.upper_bound = exp_growth
        sol = model.optimize()
        candidate_growth = float(sol.objective_value or 0.0)

    if candidate_growth - pred_growth <= _GROWTH_DELTA_THRESHOLD:
        if verbose:
            logger.info(
                "flexibilize_enz_concs: protein pool was not limiting "
                "either. Maximum growth reachable: %g", candidate_growth,
            )
        return None, None, None

    with model:
        pool_rxn.upper_bound = 1000.0
        bio_rxn.lower_bound = candidate_growth
        model.objective = POOL_EXCHANGE_ID
        model.objective_direction = "min"
        sol = model.optimize()
        new_pool = float(sol.objective_value or 0.0)

    pool_rxn.upper_bound = new_pool

    if verbose:
        logger.info(
            "flexibilize_enz_concs: prot_pool_exchange upper_bound raised "
            "from %g to %g, enabling growth %g.",
            old_pool, new_pool, candidate_growth,
        )

    return old_pool, new_pool, candidate_growth


def _build_result(
    model: "EcModel",
    proteins: list[str],
    measured_idx: np.ndarray,
    frequence: np.ndarray,
    flex_break: bool,
    exp_growth: float,
    pool_old: Optional[float],
    pool_new: Optional[float],
    bio_rxn_id: str,
    touched: Optional[np.ndarray] = None,
    families: Optional[ComplexFamilies] = None,
) -> FlexEnzResult:
    """Build the FlexEnzResult and apply the post-loop UB refinement."""
    flex_mask = frequence > 0
    if touched is not None:
        flex_mask = flex_mask | touched
    flex_indices = np.where(flex_mask)[0]
    if len(flex_indices) == 0:
        return _maybe_add_pool(FlexEnzResult(), pool_old, pool_new)

    flex_proteins = [proteins[i] for i in flex_indices]
    usage_rxn_ids = [f"{USAGE_PREFIX}{p}" for p in flex_proteins]
    old_concs_arr = np.array(
        [float(model.ec.concs[measured_idx[i]]) for i in flex_indices],
        dtype=float,
    )

    if not flex_break:
        # Refinement pass: minimize protein pool with bio_rxn in a soft
        # band around exp_growth (matching MATLAB's RAVEN setParam('var',
        # ...) call rather than pinning bio_rxn to exp_growth exactly),
        # then drop enzymes that don't actually need extra concentration.
        with model:
            rxn = model.reactions.get_by_id(bio_rxn_id)
            band = _REFINEMENT_GROWTH_BAND_VAR / 200
            rxn.lower_bound = exp_growth * (1 - band)
            rxn.upper_bound = exp_growth * (1 + band)
            model.objective = POOL_EXCHANGE_ID
            model.objective_direction = "min"
            sol = model.optimize()
            new_concs = np.array(
                [float(sol.fluxes[rid]) for rid in usage_rxn_ids],
                dtype=float,
            )

        if families is not None:
            new_concs = _per_complex(flex_proteins, old_concs_arr, new_concs, families)
        keep_old_mask = new_concs < old_concs_arr
        for i, rxn_id in enumerate(usage_rxn_ids):
            ub = (
                old_concs_arr[i] if keep_old_mask[i] else new_concs[i]
            )
            model.reactions.get_by_id(rxn_id).upper_bound = float(ub)

        survive = ~keep_old_mask
        result = FlexEnzResult(
            uniprot_ids=[
                flex_proteins[i] for i in range(len(flex_proteins)) if survive[i]
            ],
            old_concs=old_concs_arr[survive],
            flex_concs=new_concs[survive],
            frequence=frequence[flex_indices][survive],
        )
    else:
        # flex_break: take the relaxed UBs as-is.
        flex_concs = np.array(
            [
                float(model.reactions.get_by_id(rid).upper_bound)
                for rid in usage_rxn_ids
            ],
            dtype=float,
        )
        result = FlexEnzResult(
            uniprot_ids=flex_proteins,
            old_concs=old_concs_arr,
            flex_concs=flex_concs,
            frequence=frequence[flex_indices],
        )

    return _finalize_and_sort(_maybe_add_pool(result, pool_old, pool_new))


def _unit_scores(
    model: "EcModel",
    proteins: list[str],
    units: list[list[int]],
    fold_change: float,
    limit: float,
) -> np.ndarray:
    """Growth gained per fold of raising each unit's caps together.

    With duals, ``sum(reduced cost of usage_prot_i * its cap)`` over the
    unit's subunits that run at more than ``limit`` of their cap. Unlike
    the ``prot_<id>`` shadow price, the usage reaction's reduced cost
    leaves out the protein pool's price, which would otherwise be counted
    once per subunit; and the weighted sum is the same for every optimal
    dual when a levelled complex binds on all subunits at once. When no
    dual is positive (no duals exposed, or the protein pool binds and the
    solver puts the whole price on it), one solve per candidate unit with
    all its caps raised by ``fold_change`` and the pool lifted.
    """
    from .get_conc_control_coeffs import _solution_is_optimal, _solver_supports_duals

    scores = np.zeros(len(units))
    sol = model.optimize()
    if not _solution_is_optimal(sol):
        return scores
    candidates = []
    for k, members in enumerate(units):
        reactions = [model.reactions.get_by_id(f"{USAGE_PREFIX}{proteins[j]}") for j in members]
        active = [r for r in reactions if r.upper_bound > 0 and float(sol.fluxes[r.id]) / r.upper_bound > limit]
        if active:
            candidates.append((k, reactions, active))
    if _solver_supports_duals(model):
        reduced = sol.reduced_costs
        for k, _, active in candidates:
            scores[k] = sum(abs(float(reduced[r.id])) * r.upper_bound for r in active)
        scores[scores < 1e-12] = 0.0
        if (scores > 0).any():
            return scores
    # No dual singles a unit out (none exposed, or the protein pool binds as
    # well and carries the whole price): probe each candidate with the pool
    # lifted, so the probe finds the caps that limit once the pool is relaxed.
    pool_rxn = model.reactions.get_by_id(POOL_EXCHANGE_ID)
    with model:
        pool_rxn.upper_bound = max(pool_rxn.upper_bound, 1000.0)
        base = model.optimize()
        if not _solution_is_optimal(base):
            return scores
        baseline = float(base.objective_value)
        for k, reactions, _ in candidates:
            with model:
                for r in reactions:
                    r.upper_bound = r.upper_bound * fold_change
                probe = model.optimize()
            if _solution_is_optimal(probe):
                gain = float(probe.objective_value) - baseline
                scores[k] = gain / (fold_change - 1.0) if gain > _GROWTH_DELTA_THRESHOLD else 0.0
    return scores


def _per_complex(
    proteins: list[str],
    old: np.ndarray,
    new: np.ndarray,
    families: ComplexFamilies,
) -> np.ndarray:
    """One factor per complex: every flexibilized subunit takes the largest new/old of its complex."""
    out = new.copy()
    index = {p: i for i, p in enumerate(proteins)}
    seen: set[str] = set()
    for protein in proteins:
        if protein in seen:
            continue
        group = [m for m in families.members(protein) if m in index]
        seen.update(group)
        if len(group) < 2:
            continue
        ratios = [new[index[m]] / old[index[m]] for m in group if old[index[m]] > 0]
        factor = max(ratios) if ratios else 1.0
        for m in group:
            if old[index[m]] > 0:
                out[index[m]] = old[index[m]] * factor
    return out


def _maybe_add_pool(
    result: FlexEnzResult,
    pool_old: Optional[float],
    pool_new: Optional[float],
) -> FlexEnzResult:
    if pool_old is None or pool_new is None:
        return result
    result.uniprot_ids = list(result.uniprot_ids) + [_POOL_TARGET_NAME]
    result.old_concs = np.concatenate([result.old_concs, [pool_old]])
    result.flex_concs = np.concatenate([result.flex_concs, [pool_new]])
    result.frequence = np.concatenate([result.frequence, [1]])
    return result


def _finalize_and_sort(result: FlexEnzResult) -> FlexEnzResult:
    """Compute ratio_incr and sort all arrays by it descending."""
    if not result.uniprot_ids:
        return result
    ratios = np.where(
        result.old_concs > 0,
        result.flex_concs / result.old_concs,
        np.inf,
    )
    order = np.argsort(-ratios)  # descending
    return FlexEnzResult(
        uniprot_ids=[result.uniprot_ids[i] for i in order],
        old_concs=result.old_concs[order],
        flex_concs=result.flex_concs[order],
        ratio_incr=ratios[order],
        frequence=result.frequence[order],
    )
