"""Build human-readable reports of enzyme usage from an LP solution.

Ported from GECKO MATLAB:
src/geckomat/utilities/reportEnzymeUsage.m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..ec_model.pipeline.query import get_reactions_from_enzyme
from .enzyme_usage import _lookup_flux

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel
    from .enzyme_usage import EnzymeUsageResult


from ..ec_model.constants import (
    POOL_EXCHANGE_ID,
    PROT_PREFIX,
)

_FLUX_THRESHOLD = 1e-7
_PLACEHOLDER = "==="

_HIGH_CAP_COLUMNS = [
    "prot_id", "gene_id", "abs_usage", "cap_usage",
    "kcat", "source", "rxn_id", "rxn_name", "gr_rule",
]
_TOP_ABS_COLUMNS = [
    "prot_id", "gene_id", "abs_usage", "perc_usage",
    "kcat", "source", "rxn_id", "rxn_name", "gr_rule",
]


@dataclass
class EnzymeUsageReport:
    """Two human-readable tables summarising enzyme usage.

    Attributes
    ----------
    high_cap_usage
        DataFrame of enzymes whose ``cap_usage`` exceeds the
        ``high_cap_usage`` threshold. For enzymes catalysing multiple
        flux-carrying reactions, a combined header row (with
        ``"==="`` placeholders) is followed by per-reaction breakdown
        rows.
    top_abs_usage
        DataFrame of the top-N enzymes by absolute usage. Same
        layout but with ``perc_usage`` (= abs_usage / total_pool *
        100) replacing ``cap_usage``.
    total_usage_flux
        The protein pool exchange capacity (mg/gDCW) used as the
        denominator for ``perc_usage``.
    """

    high_cap_usage: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_HIGH_CAP_COLUMNS)
    )
    top_abs_usage: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_TOP_ABS_COLUMNS)
    )
    total_usage_flux: float = 0.0


def report_enzyme_usage(
    model: "EcModel",
    usage_data: "EnzymeUsageResult",
    *,
    high_cap_usage: float = 0.9,
    top_abs_usage: int = 10,
) -> EnzymeUsageReport:
    """Summarise enzyme usage as two DataFrames.

    1. **High capacity**: enzymes whose ``cap_usage`` exceeds
       ``high_cap_usage``. The enzymes catalysing more than one
       flux-carrying reaction get a combined header row plus one
       detail row per reaction, where the detail rows split the
       absolute usage proportionally to ``-S[prot_<enzyme>, rxn] *
       flux[rxn]``. Combined header rows use the literal ``"==="``
       placeholder for non-applicable string fields and ``NaN`` for
       ``kcat``.
    2. **Top absolute**: the top ``top_abs_usage`` enzymes by
       absolute usage, same shape but with ``perc_usage`` (a percent
       of ``prot_pool_exchange`` capacity) replacing ``cap_usage``.

    Ported from GECKO MATLAB:
    src/geckomat/utilities/reportEnzymeUsage.m.

    Parameters
    ----------
    model
        Full EcModel with the protein pool / usage rxn machinery.
    usage_data
        Result from ``enzyme_usage`` on the same model.
    high_cap_usage
        Capacity-usage threshold for the first report: enzymes whose
        ``cap_usage`` exceeds this fraction are included. Default
        0.9 (90% of capacity).
    top_abs_usage
        Number of top enzymes for the second report. ``0`` or
        ``inf`` returns all enzymes. Default 10.

    Returns
    -------
    EnzymeUsageReport

    Raises
    ------
    ValueError
        If ``top_abs_usage`` is negative or NaN.
    """
    if not top_abs_usage >= 0:
        raise ValueError(
            f"top_abs_usage must be a non-negative number, got {top_abs_usage!r}."
        )
    pool_rxn = model.reactions.get_by_id(POOL_EXCHANGE_ID)
    total_pool = float(pool_rxn.upper_bound)

    cap_arr = np.asarray(usage_data.cap_usage, dtype=float)
    abs_arr = np.asarray(usage_data.abs_usage, dtype=float)
    prot_ids = list(usage_data.prot_id)

    # --- High capacity ---
    high_idx = np.where(cap_arr > high_cap_usage)[0]
    high_rows: list[dict] = []
    for i in high_idx:
        prot = prot_ids[i]
        details = _per_rxn_details(model, prot, usage_data)
        if not details:
            continue
        if len(details) == 1:
            d = details[0]
            high_rows.append(_make_row(
                prot=prot, gene=_gene_for(model, prot),
                abs_usage=float(abs_arr[i]),
                cap_or_perc=float(cap_arr[i]),
                kcat=d["kcat"], source=d["source"],
                rxn_id=d["rxn_id"], rxn_name=d["rxn_name"],
                gr_rule=d["gr_rule"],
                cap_or_perc_key="cap_usage",
            ))
        else:
            # Combined header + detail rows.
            high_rows.append(_make_row(
                prot=prot, gene=_gene_for(model, prot),
                abs_usage=float(abs_arr[i]),
                cap_or_perc=float(cap_arr[i]),
                kcat=np.nan, source=_PLACEHOLDER,
                rxn_id=_PLACEHOLDER,
                rxn_name=(
                    "involved in multiple rxns, usage combined, "
                    "individual rxns below"
                ),
                gr_rule=_PLACEHOLDER,
                cap_or_perc_key="cap_usage",
            ))
            indiv_abs = np.array([d["per_rxn_abs"] for d in details])
            denom = indiv_abs.sum()
            indiv_cap = (
                (indiv_abs / denom) * float(cap_arr[i]) if denom > 0
                else np.zeros_like(indiv_abs)
            )
            for d, a, c in zip(details, indiv_abs, indiv_cap):
                high_rows.append(_make_row(
                    prot=prot, gene=_gene_for(model, prot),
                    abs_usage=float(a), cap_or_perc=float(c),
                    kcat=d["kcat"], source=d["source"],
                    rxn_id=d["rxn_id"], rxn_name=d["rxn_name"],
                    gr_rule=d["gr_rule"],
                    cap_or_perc_key="cap_usage",
                ))

    # --- Top absolute ---
    if top_abs_usage in (0, float("inf")):
        n_top = len(abs_arr)
    else:
        n_top = min(int(top_abs_usage), len(abs_arr))
    sort_idx = np.argsort(-abs_arr)[:n_top]

    top_rows: list[dict] = []
    for i in sort_idx:
        prot = prot_ids[i]
        details = _per_rxn_details(model, prot, usage_data)
        if not details:
            continue
        if len(details) == 1:
            d = details[0]
            top_rows.append(_make_row(
                prot=prot, gene=_gene_for(model, prot),
                abs_usage=float(abs_arr[i]),
                cap_or_perc=_perc(float(abs_arr[i]), total_pool),
                kcat=d["kcat"], source=d["source"],
                rxn_id=d["rxn_id"], rxn_name=d["rxn_name"],
                gr_rule=d["gr_rule"],
                cap_or_perc_key="perc_usage",
            ))
        else:
            top_rows.append(_make_row(
                prot=prot, gene=_gene_for(model, prot),
                abs_usage=float(abs_arr[i]),
                cap_or_perc=_perc(float(abs_arr[i]), total_pool),
                kcat=np.nan, source=_PLACEHOLDER,
                rxn_id=_PLACEHOLDER,
                rxn_name=(
                    "involved in multiple rxns, usage combined, "
                    "individual rxns below"
                ),
                gr_rule=_PLACEHOLDER,
                cap_or_perc_key="perc_usage",
            ))
            for d in details:
                a = d["per_rxn_abs"]
                top_rows.append(_make_row(
                    prot=prot, gene=_gene_for(model, prot),
                    abs_usage=float(a),
                    cap_or_perc=_perc(float(a), total_pool),
                    kcat=d["kcat"], source=d["source"],
                    rxn_id=d["rxn_id"], rxn_name=d["rxn_name"],
                    gr_rule=d["gr_rule"],
                    cap_or_perc_key="perc_usage",
                ))

    return EnzymeUsageReport(
        high_cap_usage=pd.DataFrame(high_rows, columns=_HIGH_CAP_COLUMNS),
        top_abs_usage=pd.DataFrame(top_rows, columns=_TOP_ABS_COLUMNS),
        total_usage_flux=total_pool,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _gene_for(model: "EcModel", protein_id: str) -> str:
    """Look up the gene ID matching a protein_id in model.ec.enzymes."""
    if protein_id in model.ec.enzymes:
        idx = model.ec.enzymes.index(protein_id)
        if idx < len(model.ec.genes):
            return model.ec.genes[idx]
    return ""


def _per_rxn_details(
    model: "EcModel",
    protein_id: str,
    usage_data: "EnzymeUsageResult",
) -> list[dict]:
    """For an enzyme, return one dict per reaction it catalyses that
    carries flux above the threshold."""
    try:
        rxns_df = get_reactions_from_enzyme(model, protein_id)
    except ValueError:
        return []

    prot_met_id = f"{PROT_PREFIX}{protein_id}"
    try:
        prot_met = model.metabolites.get_by_id(prot_met_id)
    except KeyError:
        return []

    fluxes = usage_data.fluxes
    details: list[dict] = []
    for _, row in rxns_df.iterrows():
        rxn_id = str(row["rxn_id"])
        flux = float(_lookup_flux(fluxes, rxn_id))
        # Use |flux| so reverse-running reactions are not dropped and the
        # per-reaction draws sum to the enzyme-level abs_usage header
        # (enzyme_usage aggregates with abs() too).
        if abs(flux) <= _FLUX_THRESHOLD:
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        coeff = rxn.metabolites.get(prot_met, 0.0)
        per_rxn_abs = abs(float(coeff) * flux)
        ec_idx = model.ec.rxns.index(rxn_id)
        details.append({
            "rxn_id": rxn_id,
            "rxn_name": str(row["name"]),
            "gr_rule": str(row["gpr"]),
            "kcat": float(row["kcat"]),
            "source": (
                model.ec.source[ec_idx]
                if ec_idx < len(model.ec.source) else ""
            ),
            "per_rxn_abs": per_rxn_abs,
        })
    return details


def _perc(value: float, denominator: float) -> float:
    if denominator == 0 or denominator is None:
        return 0.0
    return value / denominator * 100.0


def _make_row(
    *, prot: str, gene: str, abs_usage: float, cap_or_perc: float,
    kcat: float, source: str, rxn_id: str, rxn_name: str,
    gr_rule: str, cap_or_perc_key: str,
) -> dict:
    return {
        "prot_id": prot,
        "gene_id": gene,
        "abs_usage": abs_usage,
        cap_or_perc_key: cap_or_perc,
        "kcat": kcat,
        "source": source,
        "rxn_id": rxn_id,
        "rxn_name": rxn_name,
        "gr_rule": gr_rule,
    }
