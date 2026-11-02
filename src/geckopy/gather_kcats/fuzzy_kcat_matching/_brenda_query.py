"""BRENDA-table lookup, substrate filtering, and kcat selection.

Given an EC token (possibly with wildcards), a list of substrates,
and the BRENDA kcat/SA tables, these helpers find candidate rows,
filter them, pick a kcat, and (for the iterative version) escalate
wildcards on misses.

The six MATLAB search levels (org+subs+kcat, any+subs+kcat,
org+!subs+kcat, org+SA, any+!subs+kcat, any+SA) are tried in the
MATLAB-compat order. See the orchestrator's docstring for the
full background.

Ported from GECKO MATLAB: src/geckomat/gather_kcats/fuzzyKcatMatching.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ._escalation import escalate_wildcard
from ._organism import filter_by_organism

if TYPE_CHECKING:
    from ...databases import PhylDist


# Diffusion-limited rate cap (Bar-Even et al. 2011), 1/s. kcats
# above this are clipped, matching MATLAB.
DIFFUSION_LIMIT = 1e7

# "No match" wildcard count; a large value that never wins
# min-comparisons against real wildcard counts.
NO_MATCH_WC = 1000


# Search order. Each entry is
#   (filter_organism, filter_substrate, use_sa_table, output_origin)
# The OUTPUT origin numbering follows the MATLAB docstring:
#   1: org+subs+kcat   2: any+subs+kcat   3: org+!subs+kcat
#   4: any+!subs+kcat  5: org+SA          6: any+SA
# but the SEARCH order tries org-SA (output 5) BEFORE any-no-subs-kcat
# (output 4), matching MATLAB exactly -- see "Fix the search-order/
# output-ranking inconsistency in fuzzyKcatMatching" in
# docs/future_improvements.md.
SEARCH_LEVELS: list[tuple[bool, bool, bool, int]] = [
    (True,  True,  False, 1),
    (False, True,  False, 2),
    (True,  False, False, 3),
    (True,  False, True,  5),
    (False, False, False, 4),
    (False, False, True,  6),
]

# Position of each output origin in SEARCH_LEVELS. Selection across the EC
# tokens of one reaction compares matches by this rank, so a token matched
# at an earlier-tried level beats one matched later, as within one token.
SEARCH_RANK: dict[int, int] = {
    origin: rank for rank, (*_, origin) in enumerate(SEARCH_LEVELS)
}


def build_ec_indices(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Group BRENDA table row indices by lowercased ``ec_code``."""
    if table.empty:
        return {}
    return {
        ec_lower: np.asarray(idxs, dtype=int)
        for ec_lower, idxs in (
            table.groupby(table["ec_code"].str.lower()).indices.items()
        )
    }


def find_ec_rows(
    ec_token: str,
    ec_indices: dict[str, np.ndarray],
) -> np.ndarray:
    """Return BRENDA row indices matching ``ec_token``.

    Wildcard handling: the prefix up to the first ``-`` is matched
    case-insensitively against the start of each EC key. A fully-
    wildcarded ``-.-.-.-`` matches every row.
    """
    if not ec_indices:
        return np.array([], dtype=int)
    ec_lower = ec_token.lower()
    if "-" not in ec_token:
        return ec_indices.get(ec_lower, np.array([], dtype=int))
    prefix = ec_lower.split("-", 1)[0]
    if not prefix:
        all_idxs = list(ec_indices.values())
        return (
            np.concatenate(all_idxs) if all_idxs
            else np.array([], dtype=int)
        )
    matched = [arr for key, arr in ec_indices.items() if key.startswith(prefix)]
    return np.concatenate(matched) if matched else np.array([], dtype=int)


def filter_by_substrate(
    table: pd.DataFrame,
    rows: np.ndarray,
    substrates: list[str],
) -> np.ndarray:
    if len(rows) == 0:
        return rows
    subs_lower = {s.lower() for s in substrates if s}
    if not subs_lower:
        return np.array([], dtype=int)
    sub_col = table["substrate"].iloc[rows].str.lower()
    keep = sub_col.isin(subs_lower).values
    return rows[keep]


def _aggregate_kcats(kcats: np.ndarray, how: str) -> float:
    """Collapse the matched BRENDA kcats to a single value.

    ``"max"`` takes the highest reported turnover across the matched
    rows; ``"median"`` is more robust to assay outliers and engineered
    mutants.
    """
    if how == "median":
        return float(np.median(kcats))
    if how == "max":
        return float(np.max(kcats))
    raise ValueError(f"aggregate must be 'max' or 'median', got {how!r}")


def match_kcat(
    ec_token: str,
    substrates: list[str],
    substrate_coeffs: list[float],
    table: pd.DataFrame,
    ec_indices: dict[str, np.ndarray],
    organism: Optional[str],
    *,
    substrate_match: bool,
    sa: bool,
    phyl_dist: "PhylDist",
    org_index: Optional[int],
    aggregate: str = "max",
) -> tuple[float, int]:
    """One match attempt at one of the six search levels.

    ``substrate_match`` restricts ``table`` to rows whose substrate is
    in ``substrates`` before the organism filter runs; ``sa`` selects
    the specific-activity table instead of the kcat table. ``aggregate``
    (``"max"`` or ``"median"``) controls how the surviving rows are
    collapsed to one value.

    Returns ``(kcat_in_1_per_s, num_matched_rows)``, where ``kcat`` is
    the value picked by ``aggregate``. ``num_matched == 0`` means no
    rows survived the filters.
    """
    rows = find_ec_rows(ec_token, ec_indices)
    if len(rows) == 0:
        return 0.0, 0

    if substrate_match and not sa:
        rows = filter_by_substrate(table, rows, substrates)
        if len(rows) == 0:
            return 0.0, 0

    rows = filter_by_organism(table, rows, organism, phyl_dist, org_index)
    if len(rows) == 0:
        return 0.0, 0

    kcats = table["kcat"].iloc[rows].values

    if substrate_match and not sa:
        positive = kcats > 0
        matched_rows = rows[positive]
        kcats = kcats[positive]
        if len(kcats) == 0:
            return 0.0, 0
        # A BRENDA kcat is a turnover per molecule of the substrate it was
        # measured on, so it becomes a reaction turnover by dividing by that
        # substrate's coefficient -- the one the row matched on, not the
        # smallest coefficient in the reaction.
        coefficient = dict(zip((s.lower() for s in substrates), substrate_coeffs))
        divisors = (
            table["substrate"].iloc[matched_rows].str.lower()
            .map(coefficient).fillna(1.0).to_numpy(dtype=float)
        )
        divisors[divisors <= 0] = 1.0
        kcats = kcats / divisors

    if len(kcats) == 0:
        return 0.0, 0

    n = len(kcats)
    kcat = _aggregate_kcats(kcats, aggregate)
    if kcat > DIFFUSION_LIMIT:
        kcat = DIFFUSION_LIMIT
    return kcat, n


def main_match(
    ec_token: str,
    substrates: list[str],
    substrate_coeffs: list[float],
    brenda_kcat: pd.DataFrame,
    brenda_sa: pd.DataFrame,
    kcat_ec_indices: dict[str, np.ndarray],
    sa_ec_indices: dict[str, np.ndarray],
    organism_name: str,
    phyl_dist: "PhylDist",
    org_index: Optional[int],
    aggregate: str = "max",
) -> tuple[float, int, int]:
    """Try each of the six search levels in MATLAB's order.

    Returns ``(kcat, output_origin, num_matches)`` from the first
    level that matches. ``output_origin == 0`` means no match.
    """
    has_wildcard = "-" in ec_token
    for filter_org, filter_subs, use_sa, output_origin in SEARCH_LEVELS:
        if has_wildcard and filter_subs:
            continue
        table = brenda_sa if use_sa else brenda_kcat
        ec_idx = sa_ec_indices if use_sa else kcat_ec_indices
        org = organism_name if filter_org else None
        kcat, n = match_kcat(
            ec_token, substrates, substrate_coeffs,
            table, ec_idx, org,
            substrate_match=filter_subs,
            sa=use_sa,
            phyl_dist=phyl_dist,
            org_index=org_index,
            aggregate=aggregate,
        )
        if n > 0:
            return kcat, output_origin, n
    return 0.0, 0, 0


def iterative_match_one_ec(
    ec_token: str,
    substrates: list[str],
    substrate_coeffs: list[float],
    brenda_kcat: pd.DataFrame,
    brenda_sa: pd.DataFrame,
    kcat_ec_indices: dict[str, np.ndarray],
    sa_ec_indices: dict[str, np.ndarray],
    organism_name: str,
    phyl_dist: "PhylDist",
    org_index: Optional[int],
    aggregate: str = "max",
) -> tuple[float, int, int]:
    """Escalate wildcards in ``ec_token`` until a match is found or
    every level is wildcarded.

    Returns ``(kcat, output_origin, wildcard_level)``. On no match,
    returns ``(0.0, 0, NO_MATCH_WC)``.
    """
    current_ec = ec_token
    while True:
        wc = current_ec.count("-")
        kcat, origin, n = main_match(
            current_ec, substrates, substrate_coeffs,
            brenda_kcat, brenda_sa,
            kcat_ec_indices, sa_ec_indices,
            organism_name, phyl_dist, org_index,
            aggregate=aggregate,
        )
        if origin > 0:
            return kcat, origin, wc
        nxt = escalate_wildcard(current_ec)
        if nxt is None:
            return 0.0, 0, NO_MATCH_WC
        current_ec = nxt
