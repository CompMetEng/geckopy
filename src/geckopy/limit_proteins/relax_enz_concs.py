"""Loosen proteomics constraints by the least total amount, in one LP.

``flexibilize_enz_concs`` and ``relax_proteomics_greedy`` relax enzyme
caps one at a time: find the most limiting enzyme, raise it, re-solve,
repeat. Which enzymes end up raised then depends on how "most limiting"
is ranked and on the order in which they are found; on a large model
with noisy proteomics, small changes to that ranking can change the
result. ``relax_enz_concs`` asks the question once instead: what is the
smallest set of fold increases of the measured caps (and, if necessary,
of the protein pool) that lets the model reach the target growth rate?

Every measured enzyme's cap becomes ``usage <= conc * (1 + x)`` and the
sum of the ``x`` is minimised under ``growth >= exp_growth``. The protein
pool may grow by a fold too, at ``pool_weight`` times the price of an
enzyme's fold, so that it grows only where raising enzymes does not
suffice. With ``per_complex=True`` the measured subunits of a complex
(see ``complex_families``) share one ``x``, so concentrations levelled
by ``level_complex_concs`` stay levelled.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..ec_model.constants import POOL_EXCHANGE_ID, USAGE_PREFIX
from .flexibilize_enz_concs import (
    FlexEnzResult,
    _finalize_and_sort,
    _maybe_add_pool,
)
from .level_complex_concs import complex_families

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


def relax_enz_concs(
    model: "EcModel",
    *,
    exp_growth: Optional[float] = None,
    bio_rxn: Optional[str] = None,
    per_complex: bool = False,
    weight: str = "relative",
    pool_weight: float = 1000.0,
    floor: float = 1e-9,
) -> FlexEnzResult:
    """Relax enzyme caps by the least total amount that reaches ``exp_growth``.

    Parameters
    ----------
    model
        EcModel with ``model.ec.concs`` filled and the caps applied
        (``fill_enz_concs`` and ``constrain_enz_concs``). The usage
        upper bounds (and ``prot_pool_exchange``) are updated in place;
        ``ec.concs`` itself is left unchanged, as in
        ``flexibilize_enz_concs``.
    exp_growth
        Target growth rate. Defaults to ``params.gr_exp``.
    bio_rxn
        Biomass reaction id. Defaults to ``params.bio_rxn``.
    per_complex
        One fold per complex (its measured, owned subunits; see
        ``complex_families``) instead of one per enzyme.
    weight
        ``"relative"`` minimises the summed fold increases, so an enzyme
        measured at almost nothing is not relaxed freely because its
        absolute cost is small; ``"absolute"`` minimises the mg/gDCW
        added.
    pool_weight
        Price of one fold of the protein pool relative to one fold of an
        enzyme (or, with ``weight="absolute"``, per mg).
    floor
        Concentration below which a measured value is treated as this
        floor when scaling its fold, so a zero measurement can still be
        relaxed (at a high relative price).

    Returns
    -------
    FlexEnzResult
        The relaxed enzymes, as from ``flexibilize_enz_concs``
        (``frequence`` is 1 for every entry), with a ``"prot_pool"``
        entry when the pool was raised.

    Raises
    ------
    ValueError
        If no enzyme is measured, the weighting is unknown, or no
        relaxation reaches ``exp_growth`` (the target is then not limited
        by protein).
    """
    from optlang.symbolics import add

    from ..adapter import resolve_param

    if weight not in ("relative", "absolute"):
        raise ValueError(f"weight must be 'relative' or 'absolute', not {weight!r}")
    if exp_growth is None:
        exp_growth = resolve_param(model, None, "gr_exp", purpose="relax_enz_concs needs `exp_growth`")
        if exp_growth is None:
            raise ValueError("exp_growth not provided and params.gr_exp is unset.")
    exp_growth = float(exp_growth)
    bio = model.reactions.get_by_id(
        resolve_param(model, bio_rxn, "bio_rxn", purpose="relax_enz_concs needs the biomass reaction id")
    )

    concs = np.asarray(model.ec.concs, dtype=float)
    measured = [i for i in range(len(concs)) if np.isfinite(concs[i])]
    if not measured:
        raise ValueError("No enzyme concentrations are measured. Run fill_enz_concs and constrain_enz_concs first.")
    enzymes = list(model.ec.enzymes)

    old_caps = {i: model.reactions.get_by_id(f"{USAGE_PREFIX}{enzymes[i]}").upper_bound for i in measured}
    # folds count from the current cap, so a cap raised before is never lowered
    base = {i: min(old_caps[i], 1000.0) for i in measured}

    families = complex_families(model) if per_complex else None
    position = {e: i for i, e in enumerate(enzymes)}
    measured_set = set(measured)
    units: list[list[int]] = []
    assigned: set[int] = set()
    for i in measured:
        if i in assigned:
            continue
        if families is None or not base[i] > 0:
            unit = [i]
        else:  # a subunit measured at zero cannot share a fold with the others
            unit = [position[m] for m in families.members(enzymes[i])
                    if m in position and position[m] in measured_set
                    and position[m] not in assigned and base[position[m]] > 0]
        assigned.update(unit)
        units.append(unit)

    pool = model.reactions.get_by_id(POOL_EXCHANGE_ID)
    pool_cap = pool.upper_bound
    old_bio_bounds = bio.bounds
    previous_objective = model.objective.expression
    previous_direction = model.objective.direction

    folds, added = [], []
    for k, unit in enumerate(units):
        x = model.problem.Variable(f"relax_unit_{k}", lb=0.0, ub=None)
        added.append(x)
        for i in unit:
            rxn = model.reactions.get_by_id(f"{USAGE_PREFIX}{enzymes[i]}")
            rxn.upper_bound = 1000.0
            added.append(model.problem.Constraint(
                rxn.flux_expression - max(base[i], floor) * x, ub=base[i], name=f"relax_cap_{enzymes[i]}",
            ))
        folds.append((x, unit))
    pool_x = model.problem.Variable("relax_pool", lb=0.0, ub=None)
    pool.upper_bound = 1000.0
    added += [pool_x, model.problem.Constraint(pool.flux_expression - pool_cap * pool_x, ub=pool_cap, name="relax_cap_pool")]
    bio.bounds = (exp_growth, max(bio.upper_bound, exp_growth))
    model.add_cons_vars(added)

    try:
        terms = [(1.0 if weight == "relative" else sum(max(base[i], floor) for i in unit)) * x for x, unit in folds]
        terms.append(pool_weight * (1.0 if weight == "relative" else pool_cap) * pool_x)
        model.objective = model.problem.Objective(add(terms), direction="min")
        solution = model.optimize()
        feasible = solution.status == "optimal"
        values = model.solver.primal_values if feasible else {}
    finally:
        model.remove_cons_vars(added)
        model.objective = model.problem.Objective(previous_objective, direction=previous_direction)
        bio.bounds = (old_bio_bounds[0], max(old_bio_bounds[1], exp_growth))

    if not feasible:
        for i, cap in old_caps.items():
            model.reactions.get_by_id(f"{USAGE_PREFIX}{enzymes[i]}").upper_bound = cap
        pool.upper_bound = pool_cap
        raise ValueError(
            f"relax_enz_concs: no relaxation of the measured enzymes or the protein pool reaches "
            f"growth {exp_growth:g} ({solution.status}); the target is not limited by protein."
        )

    ids, old, new = [], [], []
    for x, unit in folds:
        fold = float(values.get(x.name, 0.0))
        for i in unit:
            cap = base[i] + max(base[i], floor) * fold
            model.reactions.get_by_id(f"{USAGE_PREFIX}{enzymes[i]}").upper_bound = cap if fold > 1e-9 else old_caps[i]
            if fold > 1e-9:
                ids.append(enzymes[i]); old.append(base[i]); new.append(cap)
    pool_fold = float(values.get(pool_x.name, 0.0))
    pool_new = pool_cap * (1 + pool_fold) if pool_fold > 1e-9 else None
    pool.upper_bound = pool_new if pool_new is not None else pool_cap
    logger.info("relax_enz_concs: %d enzymes relaxed%s", len(ids),
                f", protein pool {pool_cap:g} -> {pool_new:g}" if pool_new is not None else "")
    result = FlexEnzResult(uniprot_ids=ids, old_concs=np.array(old, dtype=float),
                           flex_concs=np.array(new, dtype=float), frequence=np.ones(len(ids), dtype=int))
    return _finalize_and_sort(_maybe_add_pool(result, pool_cap if pool_new is not None else None, pool_new))
