"""Greedy shadow-price-ordered relaxation of proteomics constraints.

Ported from the legacy geckopy package described in Carrasco et al.
(2023, https://doi.org/10.1128/spectrum.01705-23), file
geckopy/experimental/relaxation.py
(relax_proteomics_greedy), adapted to the new substrate.

Algorithm: while the model can't reach ``minimal_growth``, find
the enzyme with the largest absolute shadow price among those
still proteomics-constrained, relax it (set usage upper bound
back to the unconstrained default), re-solve, repeat. Returns
the set of relaxed enzymes and a step-by-step trace.

Different relaxation strategy from ``flexibilize_enz_concs``: this
one is shadow-price-ordered (purely LP-driven, one solve per step)
and often converges in fewer iterations when one or two enzymes
dominate the infeasibility. The trade-off is that it fully
unconstrains each picked enzyme with no tighten-back pass, so the
result is looser (less proteomics-faithful) than
``flexibilize_enz_concs``. See ``docs/relaxation_methods.md`` for
the full comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..solver_support import constraint_shadow_prices

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


@dataclass
class RelaxationStep:
    iteration: int
    relaxed_uniprot: str
    growth_before: float
    growth_after: float
    shadow_price: float


@dataclass
class GreedyRelaxResult:
    relaxed: dict[str, float]
    trace: list[RelaxationStep] = field(default_factory=list)
    final_growth: float = 0.0
    converged: bool = False


def relax_proteomics_greedy(
    model: "EcModel",
    *,
    minimal_growth: float,
    enzyme_set: Optional[set[str]] = None,
    max_iterations: int = 100,
    default_upper_bound: float = 1000.0,
) -> GreedyRelaxResult:
    """Greedy relaxation of proteomics constraints until growth >= minimal_growth.

    Parameters
    ----------
    model
        EcModel with proteomics-constrained enzymes (some
        ``usage_prot_<id>`` reactions have finite upper bounds
        from ``constrain_enz_concs``).
    minimal_growth
        Target objective value.
    enzyme_set
        Subset of uniprot IDs eligible for relaxation. ``None`` =
        all currently proteomics-constrained enzymes.
    max_iterations
        Safety cap; raises ``RuntimeError`` if exceeded without
        convergence.
    default_upper_bound
        Value to restore on a relaxed usage reaction. Matches the
        default set by ``add_protein_usage_reactions``.

    Returns
    -------
    GreedyRelaxResult
        ``relaxed`` maps uniprot to the original concentration
        (so the caller can restore if desired); ``trace`` records
        each step; ``final_growth`` is the objective value reached
        at the last iteration attempted; ``converged`` is True iff
        ``final_growth >= minimal_growth``.

    Ported from the legacy geckopy package (Carrasco et al., 2023,
    https://doi.org/10.1128/spectrum.01705-23),
    geckopy/experimental/relaxation.py (relax_proteomics_greedy).
    """
    candidates = _eligible_enzymes(model, enzyme_set, default_upper_bound)
    relaxed: dict[str, float] = {}
    trace: list[RelaxationStep] = []

    growth = model.slim_optimize()
    if np.isnan(growth):
        growth = -np.inf

    for it in range(max_iterations):
        if growth >= minimal_growth:
            return GreedyRelaxResult(
                relaxed=relaxed, trace=trace,
                final_growth=float(growth), converged=True,
            )
        if not candidates:
            logger.warning("No more candidates to relax; stopping")
            return GreedyRelaxResult(
                relaxed=relaxed, trace=trace,
                final_growth=float(growth), converged=False,
            )

        ordered = sorted(candidates)
        prices = dict(zip(
            ordered,
            constraint_shadow_prices(model, [model.enzymes.get_by_id(u).prot_metabolite_id for u in ordered]),
        ))
        target = max(ordered, key=lambda u: abs(prices[u]))
        enz = model.enzymes.get_by_id(target)
        original_conc = enz.concentration
        sp = prices[target]

        enz.upper_bound = default_upper_bound
        relaxed[target] = original_conc
        candidates.discard(target)

        new_growth = model.slim_optimize()
        if np.isnan(new_growth):
            new_growth = -np.inf
        trace.append(RelaxationStep(
            iteration=it,
            relaxed_uniprot=target,
            growth_before=float(growth),
            growth_after=float(new_growth),
            shadow_price=float(sp),
        ))
        growth = new_growth

    raise RuntimeError(
        f"Did not converge in {max_iterations} iterations "
        f"(final growth {growth:.4g} < target {minimal_growth:.4g})"
    )


def _eligible_enzymes(
    model: "EcModel",
    enzyme_set: Optional[set[str]],
    default_upper_bound: float,
) -> set[str]:
    """Enzymes currently proteomics-constrained (ub < default)."""
    candidates: set[str] = set()
    pool = model.ec.enzymes if enzyme_set is None else enzyme_set
    for u in pool:
        if u not in model.ec.enzymes:
            continue
        enz = model.enzymes.get_by_id(u)
        if enz.upper_bound < default_upper_bound:
            candidates.add(u)
    return candidates
