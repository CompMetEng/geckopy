"""Parsimonious FBA that minimises total enzyme usage.

Standard FBA finds *some* flux distribution that maximises the
objective. Parsimonious FBA (pFBA) goes one step further: among
all flux distributions that achieve the maximum, it picks the one
with the smallest total flux. This is the classical pFBA from
cobra-py — it doesn't know about enzymes.

``pfba_enzymes`` is the enzyme-aware version: among all
biomass-optimal flux distributions, it picks the one that uses the
least enzyme mass (the smallest sum of ``usage_prot_*`` fluxes,
weighted by the cost coefficients in the LP). For ecModels this is
usually what you want, because it picks the most parsimonious
*proteome*, not just the most parsimonious *fluxes*.

Ported from the legacy geckopy package described in Carrasco et al.
(2023, https://doi.org/10.1128/spectrum.01705-23), file
geckopy/flux_analysis.py:342-386 (pfba_protein), adapted to the new
substrate (``usage_prot_<id>`` reactions instead of a Protein DictList).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import cobra
from optlang.symbolics import Zero

from ..ec_model.constants import USAGE_PREFIX

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


_PFBA_OBJ_NAME = "_pfba_enzymes_objective"

# RAVEN's solveLP multiplies the fraction of the optimum by this margin in the
# flux-minimising stage (getPfbaEnzymes calls it with minFlux 1). A solver reports the
# optimum to within its feasibility tolerance, so a second solve that demands the full
# value can be infeasible.
_PFBA_MARGIN = 0.999999


def pfba_enzymes(
    model: "EcModel",
    objective: Union[cobra.Reaction, str, None] = None,
    *,
    fraction_of_optimum: float = 1.0,
) -> cobra.Solution:
    """Parsimonious FBA minimising the L1 norm of ``usage_prot_*`` fluxes.

    Sets the objective (if given), fixes it at
    ``fraction_of_optimum * 0.999999 * optimal_value``, then minimises
    ``sum(usage_prot_*.forward_variable)``. Usage reactions are
    forward-only in the GECKO 3 layout (``lb=0``); if a user has
    flipped one to allow reverse flux, its ``reverse_variable`` is
    included automatically.

    Ported from the legacy geckopy package (Carrasco et al., 2023,
    https://doi.org/10.1128/spectrum.01705-23),
    geckopy/flux_analysis.py:342-386.

    Parameters
    ----------
    model
        Full EcModel (not gecko-light) with ``usage_prot_*``
        reactions installed.
    objective
        Reaction (or reaction id) to optimise before minimising
        enzyme usage. If ``None``, the model's current objective is
        used as-is.
    fraction_of_optimum
        Fraction of the objective's optimal value to fix as a
        constraint before minimising enzyme usage (default 1.0). The
        fraction is multiplied by 0.999999, the margin of RAVEN's
        ``solveLP`` that MATLAB's ``getPfbaEnzymes`` applies, so the
        default fixes the objective at 0.999999 of its optimum and
        0.9 fixes it at 0.8999991. A lower value trades some
        objective value for potentially lower enzyme usage.

    Returns
    -------
    cobra.Solution
        Standard cobra solution over all reactions, at the flux
        distribution that minimises total (mass-weighted) enzyme
        usage subject to the objective constraint.

    Raises
    ------
    NotImplementedError
        If ``model.ec.gecko_light`` is True.
    ValueError
        If ``model``'s objective is already the internal
        ``pfba_enzymes`` objective, indicating a nested call.
    cobra.exceptions.OptimizationError
        If minimising enzyme usage is infeasible at the requested
        ``fraction_of_optimum``.
    """
    if model.ec.gecko_light:
        raise NotImplementedError(
            "pfba_enzymes requires usage reactions (not gecko-light)"
        )

    with model as m:
        if objective is not None:
            m.objective = objective
        if m.objective.name == _PFBA_OBJ_NAME:
            raise ValueError(
                f"Objective is already {_PFBA_OBJ_NAME}; "
                "pfba_enzymes appears to be nested."
            )
        cobra.util.fix_objective_as_constraint(
            m, fraction=fraction_of_optimum * _PFBA_MARGIN,
        )

        usage_rxns = [r for r in m.reactions if r.id.startswith(USAGE_PREFIX)]
        variables = []
        for r in usage_rxns:
            variables.append(r.forward_variable)
            if r.lower_bound < 0:
                variables.append(r.reverse_variable)

        m.objective = m.problem.Objective(
            Zero, direction="min", sloppy=True, name=_PFBA_OBJ_NAME,
        )
        m.objective.set_linear_coefficients({v: 1.0 for v in variables})
        # Usage fluxes are already mass-scaled (the catalysed-reaction
        # coupling carries MW/(kcat*3600) and prot_pool is in mg/gDW), so a
        # unit-weighted sum minimises total enzyme mass.
        if m.slim_optimize(error_value=None) is None:
            raise cobra.exceptions.OptimizationError(
                "pfba_enzymes: minimising enzyme usage was infeasible at the "
                f"requested fraction_of_optimum ({fraction_of_optimum})."
            )
        solution = cobra.core.solution.get_solution(m)
    return solution
