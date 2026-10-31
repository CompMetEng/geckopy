"""Compute enzyme-concentration control coefficients via LP sensitivity.

Ported from GECKO MATLAB:
src/geckomat/limit_proteins/getConcControlCoeffs.m.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


from ..solver_support import optimize_model
from ..ec_model.constants import USAGE_PREFIX

logger = logging.getLogger(__name__)

_GROWTH_DELTA_THRESHOLD = 1e-10

# Relative raise of a usage upper bound that confirms a nonzero reduced cost (0.01 percent).
_VERIFY_STEP = 1e-4

# A reduced cost is kept when the optimum rises by at least this fraction of the rise the
# reduced cost predicts for the step (reduced cost times the bound times ``_VERIFY_STEP``).
_VERIFY_MIN_FRACTION = 0.5

# Smallest rise of the optimum that counts, relative to max(1, |optimum|); below it the
# two solves do not differ beyond solver noise.
_VERIFY_GROWTH_FLOOR = 1e-12

# Solvers we have explicitly confirmed do not expose LP duals at all. Empty: optlang's scipy
# backend raises on ``dual`` but keeps HiGHS's bound marginals (see
# :func:`_scipy_upper_marginal`), and any solver whose dual read fails or returns NaN falls
# back to the finite-difference loop anyway.
_NO_DUALS_SOLVER_TAGS: tuple[str, ...] = ()

# One-shot info reporting when the shadow-price path is unavailable.
_FALLBACK_REPORTED: set[str] = set()


def get_conc_control_coeffs(
    model: "EcModel",
    proteins: Optional[list[str]] = None,
    fold_change: float = 2.0,
    limit: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute control coefficients for enzyme concentrations.

    For each protein in ``proteins``, the coefficient quantifies how
    much the LP optimum (typically growth) increases per unit increase
    in that enzyme's upper bound. Proteins whose usage flux is at most
    ``limit`` times their upper bound are skipped (no sensitivity to
    expect there).

    The default path reads, from a single LP solve, the reduced cost of
    each analysed usage reaction's forward variable. That is the local
    marginal d(objective)/d(usage_ub) of the enzyme's own bound: it is
    zero for an enzyme whose bound does not bind, also when the shared
    protein pool does, because the pool's price is part of the
    ``prot_<id>`` metabolite dual and cancels out of the reduced cost
    (``|dual(prot_<id>) - dual(prot_pool)|``). A nonzero reduced cost is
    kept only if raising that bound by 0.01 percent raises the optimum
    by at least half of the rise the reduced cost predicts (one extra
    solve per nonzero value), because the dual overestimates the effect
    at a degenerate vertex. The test is relative to the predicted rise,
    so a small coefficient on a small bound is kept as readily as a large
    one; the rise must also exceed 1e-12 times max(1, |optimum|). A zero
    reduced cost needs no check. With optlang's scipy backend (HiGHS) the reduced cost is
    read from the bound marginal of the last ``linprog`` result, since optlang does not
    expose variable duals there. When the active solver gives no usable variable duals
    (a missing or NaN value, or a scipy result without bound marginals) the function falls
    back to the per-protein finite-difference loop, which needs one solve per protein
    (several minutes on a genome-scale ecModel), and logs once that it has done so. The
    fold-change probe size is only used in that fallback.

    Ported from GECKO MATLAB:
    src/geckomat/limit_proteins/getConcControlCoeffs.m.

    Parameters
    ----------
    model
        EcModel with a defined LP objective (typically a biomass
        reaction) and ``usage_prot_<enzyme>`` reactions for every
        analysed protein.
    proteins
        UniProt IDs to analyse. Defaults to ``model.ec.enzymes``.
    fold_change
        Probe size for the finite-difference fallback (default 2.0).
        Ignored on the shadow-price path.
    limit
        Skip proteins whose usage flux is at most ``limit`` times
        their upper bound. Default 0.0 includes any protein with
        non-zero flux.

    Returns
    -------
    enz_mask
        Boolean array of length ``len(proteins)``; True where the
        protein was analysed.
    control_coeffs
        Float array of length ``len(proteins)``; 0.0 where the
        protein was either skipped or showed no sensitivity.
    """
    if proteins is None:
        proteins = list(model.ec.enzymes)

    n = len(proteins)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    if _solver_supports_duals(model):
        return _shadow_price_coeffs(model, proteins, limit, fold_change)
    _report_finite_difference_fallback(model, "does not expose LP duals")
    return _finite_difference_coeffs(model, proteins, fold_change, limit)


def _shadow_price_coeffs(
    model: "EcModel", proteins: list[str], limit: float,
    fold_change: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-solve control coefficients from the usage reactions' reduced costs.

    Falls back to :func:`_finite_difference_coeffs` when the solver gives
    no usable duals.
    """
    n = len(proteins)
    enz_mask = np.zeros(n, dtype=bool)
    coeffs = np.zeros(n, dtype=float)

    sol = optimize_model(model)
    if not _solution_is_optimal(sol):
        return enz_mask, coeffs
    initial_growth = sol.objective_value
    fluxes = sol.fluxes

    cobra_rxn_ids = {r.id for r in model.reactions}
    reduced_costs: dict[int, float] = {}
    pinned_at_lower: set[int] = set()
    for i, protein in enumerate(proteins):
        rxn_id = f"{USAGE_PREFIX}{protein}"
        if rxn_id not in cobra_rxn_ids:
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        prev_ub = rxn.upper_bound
        if prev_ub <= 0:
            continue
        usage_flux = float(fluxes.get(rxn_id, 0.0))
        if usage_flux / prev_ub <= limit:
            continue
        enz_mask[i] = True
        rc = _forward_reduced_cost(rxn, usage_flux)
        if rc is None:
            _report_finite_difference_fallback(model, "returns no usable variable duals")
            return _finite_difference_coeffs(model, proteins, fold_change, limit)
        reduced_costs[i] = rc
        if rxn.lower_bound > 0 and usage_flux <= rxn.lower_bound * (1.0 + 1e-9):
            # Held at its lower bound: raising the upper bound changes nothing.
            pinned_at_lower.add(i)

    for i, rc in reduced_costs.items():
        if rc == 0.0 or i in pinned_at_lower:
            continue
        rxn = model.reactions.get_by_id(f"{USAGE_PREFIX}{proteins[i]}")
        predicted_gain = rc * rxn.upper_bound * _VERIFY_STEP
        with model:
            rxn.upper_bound = rxn.upper_bound * (1.0 + _VERIFY_STEP)
            new_sol = optimize_model(model)
        if not _solution_is_optimal(new_sol):
            continue
        gain = new_sol.objective_value - initial_growth
        if gain >= max(
            _VERIFY_MIN_FRACTION * predicted_gain,
            _VERIFY_GROWTH_FLOOR * max(1.0, abs(initial_growth)),
        ):
            coeffs[i] = rc

    return enz_mask, coeffs


def _forward_reduced_cost(rxn, flux: Optional[float] = None) -> Optional[float]:
    """|reduced cost| of the reaction's forward variable, or None if unavailable.

    ``flux`` is the reaction's flux in the solution the reduced cost must belong to; the
    scipy path needs it to confirm that the stored result is that solution.

    ``reaction.reduced_cost`` is the forward variable's value minus the reverse
    variable's, and the reverse variable of a usage reaction carries the opposite
    sign, so it is twice the sensitivity to the upper bound; the forward variable
    alone is that sensitivity.
    """
    try:
        value = float(rxn.forward_variable.dual)
    except NotImplementedError:
        value = _scipy_upper_marginal(rxn, flux)
        if value is None:
            return None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if not np.isfinite(value):
        return None
    return abs(value)


def _scipy_upper_marginal(rxn, flux: Optional[float]) -> Optional[float]:
    """Marginal of the forward variable's upper bound in optlang's scipy backend, or None.

    optlang raises ``NotImplementedError`` for ``Variable.dual`` with the scipy solver but
    keeps the whole ``linprog`` result in ``problem._solution``. For HiGHS that result holds
    ``upper.marginals``: d(objective)/d(upper bound) of the minimised problem, in the order
    of the variables handed to ``linprog``. ``cobra.Model.optimize`` resets optlang's own copy
    of the primal values after the solve (it restores the objective direction), so the stored
    result is checked against ``flux``, the flux cobra returned for this reaction: the forward
    minus the reverse variable of the stored result must equal it. Without ``flux``, or when
    the values differ, the result is not used (None) and the finite-difference loop runs.
    """
    if flux is None:
        return None
    try:
        problem = rxn.model.solver.problem
        solution = problem._solution
        marginals = solution.upper.marginals
        names = list(problem.bounds)
        forward = names.index(rxn.forward_variable.name)
        reverse = names.index(rxn.reverse_variable.name)
        stored = float(solution.x[forward]) - float(solution.x[reverse])
        if abs(stored - flux) > 1e-9 * max(1.0, abs(flux)):
            return None
        return float(marginals[forward])
    except (AttributeError, IndexError, ValueError, TypeError, KeyError):
        return None


def _finite_difference_coeffs(
    model: "EcModel",
    proteins: list[str],
    fold_change: float,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein finite-difference fallback (one solve per protein).

    Same contract as :func:`get_conc_control_coeffs`. Used when the solver
    does not expose usable variable duals.
    """
    n = len(proteins)
    enz_mask = np.zeros(n, dtype=bool)
    coeffs = np.zeros(n, dtype=float)

    initial_sol = optimize_model(model)
    if not _solution_is_optimal(initial_sol):
        return enz_mask, coeffs
    initial_growth = initial_sol.objective_value
    initial_fluxes = initial_sol.fluxes

    cobra_rxn_ids = {r.id for r in model.reactions}
    for i, protein in enumerate(proteins):
        rxn_id = f"{USAGE_PREFIX}{protein}"
        if rxn_id not in cobra_rxn_ids:
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        prev_ub = rxn.upper_bound
        if prev_ub <= 0:
            continue
        usage_flux = float(initial_fluxes.get(rxn_id, 0.0))
        if usage_flux / prev_ub <= limit:
            continue
        enz_mask[i] = True
        new_ub = prev_ub * fold_change
        with model:
            rxn.upper_bound = new_ub
            new_sol = optimize_model(model)
        if not _solution_is_optimal(new_sol):
            continue
        new_growth = new_sol.objective_value
        delta_growth = new_growth - initial_growth
        if delta_growth > _GROWTH_DELTA_THRESHOLD:
            coeffs[i] = delta_growth / (new_ub - prev_ub)

    return enz_mask, coeffs


def _solution_is_optimal(sol) -> bool:
    """True iff ``sol`` is a genuine LP optimum with a usable objective.

    Requires ``status == "optimal"``; a non-NaN ``objective_value`` on
    its own is not sufficient, since on an infeasible LP some solvers
    (notably glpk via optlang) can still report a non-NaN
    ``objective_value`` (e.g. the bound that made it infeasible) with
    ``fluxes`` populated from their last attempted basis. Any
    non-optimal state (infeasible, unbounded, suboptimal, ...) is
    rejected.
    """
    if sol is None or sol.status != "optimal":
        return False
    obj = sol.objective_value
    return obj is not None and not np.isnan(obj)


def _solver_supports_duals(model: "EcModel") -> bool:
    """True unless the solver is listed in ``_NO_DUALS_SOLVER_TAGS``.

    glpk/glpk_exact/gurobi/cplex expose variable duals for LP optima and optlang's scipy
    backend exposes them through :func:`_scipy_upper_marginal`. A solver whose dual read
    fails is detected per value in :func:`_forward_reduced_cost`, which sends the call to the
    finite-difference loop.
    """
    module = type(model.solver).__module__.lower()
    return not any(tag in module for tag in _NO_DUALS_SOLVER_TAGS)


def _report_finite_difference_fallback(model: "EcModel", reason: str) -> None:
    """Log once per solver that we fell back to the per-protein loop."""
    module = type(model.solver).__module__
    if module in _FALLBACK_REPORTED:
        return
    _FALLBACK_REPORTED.add(module)
    logger.info(
        "get_conc_control_coeffs: solver %r %s; "
        "using the per-protein finite-difference fallback (one solve per "
        "protein, nx slower). This is reported once per process.",
        module, reason,
    )
