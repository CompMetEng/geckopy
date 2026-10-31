"""Find the enzymes that limit the current objective.

When an ecModel is solved, some enzymes are used up to their upper bound and some have spare capacity. An enzyme that is not used up cannot limit the objective: raising its bound does not change the optimum. An enzyme that is used up may limit the objective, but it does not have to, because another capacity constraint can be active at the same optimum.

``get_enzyme_bottlenecks`` ranks the enzymes by the re-solved change of the objective when the upper bound of one enzyme is raised by a factor ``fold_change``. Only the enzymes at their bound are re-solved; every other enzyme has a change of 0 without a solve. The LP shadow price of the ``prot_<id>`` constraint is reported for reference and is not used for the order: at a degenerate optimum, which enzyme-constrained models routinely have, several constraints are active and the solver assigns the dual to one of them, so the dual depends on the basis and not on the model.

Ported from the legacy geckopy package described in Carrasco et al. (2023, https://doi.org/10.1128/spectrum.01705-23), file geckopy/flux_analysis.py:275-281 (get_protein_bottlenecks).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

from ..ec_model.constants import POOL_EXCHANGE_ID
from ..solver_support import constraint_shadow_prices, optimize_model

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

# An enzyme is at its bound when |flux| / upper_bound is at least 1 - _AT_CAP_TOL.
_AT_CAP_TOL = 1e-6
# Changes of the objective below this value are set to 0.
_GAIN_TOL = 1e-10


def get_enzyme_bottlenecks(
    model: "EcModel", *, top: int = 10, fold_change: float = 2.0
) -> pd.DataFrame:
    """Top-N enzymes by the change of the objective when their bound is raised.

    Solves the model. An enzyme is at its bound when its upper bound is positive and ``|flux| / upper_bound`` is at least ``1 - 1e-6``. For each such enzyme the upper bound of its usage reaction is raised to ``upper_bound * fold_change``, the model is solved again and ``objective_gain`` is the new objective minus the objective of the first solve (0 when the new solve is not optimal or the change is below 1e-10). The model is restored after each solve. Every other enzyme has ``objective_gain`` 0 without a solve, because raising a bound that does not bind cannot change the optimum. The change is signed in the direction of the objective as the solver returns it, so it is positive for a maximised objective that increases.

    The rows are sorted by ``objective_gain`` descending, compared at 9 decimals, and ties keep the order of ``model.enzymes``. ``shadow_price`` (the dual of the ``prot_<id>`` constraint) is reported for reference and does not enter the order, because at a degenerate optimum it depends on the basis the solver ended in.

    ``multiple_active`` is True for an enzyme when a capacity constraint other than its own is active at the same optimum, which is where the dual says little about the enzyme. The capacity constraints counted are the enzymes at their bound and the protein pool exchange (``prot_pool_exchange``) when its flux is within 1e-6 relative of a finite, positive upper bound.

    Ported from the legacy geckopy package (Carrasco et al., 2023, https://doi.org/10.1128/spectrum.01705-23), geckopy/flux_analysis.py:275-281.

    Parameters
    ----------
    model
        EcModel to solve.
    top
        Number of top-ranked enzymes to return. Default 10.
    fold_change
        Factor by which the upper bound of an enzyme at its bound is raised. Default 2.0.

    Returns
    -------
    pandas.DataFrame
        Indexed by uniprot id, columns ``[gene, objective_gain, shadow_price, flux, cap_usage, upper_bound, multiple_active]``, truncated to ``top`` rows. ``df.attrs["n_resolved"]`` is the number of enzymes that were solved again and ``df.attrs["active_constraints"]`` the number of active capacity constraints.

    Raises
    ------
    RuntimeError
        If the solve status is not optimal.
    """
    sol = optimize_model(model)
    if sol.status != "optimal":
        raise RuntimeError(
            f"Solver status {sol.status!r}; cannot rank bottlenecks"
        )
    base = sol.objective_value

    enzymes = list(model.enzymes)
    prices = constraint_shadow_prices(model, [enz.prot_metabolite_id for enz in enzymes])
    rows = []
    for enz, price in zip(enzymes, prices):
        rows.append({
            "uniprot": enz.id,
            "gene": enz.gene,
            "objective_gain": 0.0,
            "shadow_price": float(price),
            "flux": enz.flux,
            "cap_usage": enz.cap_usage,
            "upper_bound": enz.upper_bound,
        })

    at_cap = [
        i for i, row in enumerate(rows)
        if row["upper_bound"] > 0 and row["cap_usage"] >= 1 - _AT_CAP_TOL
    ]
    active = len(at_cap)
    if model.reactions.has_id(POOL_EXCHANGE_ID):
        pool = model.reactions.get_by_id(POOL_EXCHANGE_ID)
        pool_ub = float(pool.upper_bound)
        if 0 < pool_ub < math.inf and abs(float(pool.flux)) >= pool_ub * (1 - _AT_CAP_TOL):
            active += 1

    for i in at_cap:
        usage = enzymes[i].usage_reaction
        with model:
            usage.upper_bound = rows[i]["upper_bound"] * fold_change
            new = optimize_model(model)
            if new.status == "optimal":
                gain = new.objective_value - base
                rows[i]["objective_gain"] = gain if gain >= _GAIN_TOL else 0.0
    if at_cap:
        optimize_model(model)

    at_cap_set = set(at_cap)
    for i, row in enumerate(rows):
        row["multiple_active"] = (active - (1 if i in at_cap_set else 0)) >= 1

    rows.sort(key=lambda row: -round(row["objective_gain"], 9))
    df = pd.DataFrame(rows).set_index("uniprot")
    df = df[[
        "gene", "objective_gain", "shadow_price", "flux", "cap_usage",
        "upper_bound", "multiple_active",
    ]]
    df = df.head(top)
    df.attrs["n_resolved"] = len(at_cap)
    df.attrs["active_constraints"] = active
    return df
