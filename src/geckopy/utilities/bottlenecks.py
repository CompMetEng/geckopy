"""Find the enzymes that are limiting the current objective.

When you solve an ecModel, some enzymes are saturated and some
have spare capacity. The saturated ones are the ones whose
removal (or whose increased availability) would shift the
objective — they're the *bottlenecks*. The LP's shadow price for
each ``prot_<id>`` mass-balance constraint quantifies that
sensitivity directly.

``get_enzyme_bottlenecks`` runs the solve, ranks every enzyme by
absolute shadow price, and returns the top N as a DataFrame. The
top row is the enzyme whose extra availability would help the
objective the most.

Ported from the legacy geckopy package described in Carrasco et al.
(2023, https://doi.org/10.1128/spectrum.01705-23), file
geckopy/flux_analysis.py:275-281 (get_protein_bottlenecks).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..solver_support import constraint_shadow_prices, optimize_model

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


def get_enzyme_bottlenecks(model: "EcModel", *, top: int = 10) -> pd.DataFrame:
    """Top-N enzymes by absolute shadow price.

    Solves the model, ranks enzymes by ``|shadow_price|``, returns a
    DataFrame indexed by uniprot id with columns
    ``[gene, shadow_price, flux, cap_usage, upper_bound]``.

    Ported from the legacy geckopy package (Carrasco et al., 2023,
    https://doi.org/10.1128/spectrum.01705-23),
    geckopy/flux_analysis.py:275-281.

    Parameters
    ----------
    model
        EcModel to solve.
    top
        Number of top-ranked enzymes to return. Default 10.

    Returns
    -------
    pandas.DataFrame
        Indexed by uniprot id, columns
        ``[gene, shadow_price, flux, cap_usage, upper_bound]``,
        sorted by descending ``|shadow_price|`` and truncated to
        ``top`` rows.

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

    enzymes = list(model.enzymes)
    prices = constraint_shadow_prices(model, [enz.prot_metabolite_id for enz in enzymes])
    rows = []
    for enz, price in zip(enzymes, prices):
        rows.append({
            "uniprot": enz.id,
            "gene": enz.gene,
            "shadow_price": float(price),
            "flux": enz.flux,
            "cap_usage": enz.cap_usage,
            "upper_bound": enz.upper_bound,
        })
    df = pd.DataFrame(rows).set_index("uniprot")
    df = df.reindex(
        df["shadow_price"].abs().sort_values(ascending=False).index
    )
    return df.head(top)
