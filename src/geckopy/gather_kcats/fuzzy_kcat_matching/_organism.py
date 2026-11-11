"""Organism resolution and per-row organism filtering against PhylDist.

The model organism's name (from the adapter's ``params.org_name``)
is looked up in the KEGG phylogenetic-distance struct. When BRENDA
has no exact organism match, the matcher uses this struct to keep
the rows for the phylogenetically closest organism(s).

Ported from GECKO MATLAB: src/gather_kcats/fuzzyKcatMatching.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...databases import PhylDist


def resolve_organism_index(
    org_name: str, phyl_dist: "PhylDist",
) -> Optional[int]:
    """Find the model organism's row in PhylDist.

    Returns ``None`` if neither a direct match nor a genus fallback
    is found (in which case organism filtering is skipped, matching
    MATLAB).
    """
    if not org_name:
        return None
    org_lower = org_name.lower()
    direct = phyl_dist.name_to_index.get(org_lower)
    if direct is not None:
        return direct
    parts = org_lower.split(None, 1)
    if parts:
        genus_indices = phyl_dist.genus_to_indices.get(parts[0], [])
        if genus_indices:
            return genus_indices[0]
    return None


def filter_by_organism(
    table: pd.DataFrame,
    rows: np.ndarray,
    organism: Optional[str],
    phyl_dist: "PhylDist",
    org_index: Optional[int],
) -> np.ndarray:
    """Filter ``rows`` of ``table`` by organism.

    If ``organism`` is given, an exact case-insensitive match is
    required. Otherwise the rows are restricted to the
    phylogenetically closest organism(s) per ``phyl_dist`` (using
    ``org_index`` as the model-organism row). If ``org_index`` is
    also ``None``, no filtering is applied -- matches MATLAB when
    the model organism has no KEGG entry.
    """
    if len(rows) == 0:
        return rows

    if organism is not None and organism != "":
        org_col = table["organism"].iloc[rows].str.lower()
        keep = org_col.values == organism.lower()
        return rows[keep]

    if org_index is None:
        return rows

    organisms = table["organism"].iloc[rows].values
    valid_kegg: list[int] = []
    valid_rows: list[int] = []
    for r, org in zip(rows, organisms):
        ol = str(org).lower()
        kegg_idx = phyl_dist.name_to_index.get(ol)
        if kegg_idx is None:
            parts = ol.split(None, 1)
            if parts:
                gen_indices = phyl_dist.genus_to_indices.get(parts[0], [])
                if gen_indices:
                    kegg_idx = gen_indices[0]
        if kegg_idx is not None:
            valid_kegg.append(kegg_idx)
            valid_rows.append(int(r))

    if not valid_rows:
        return np.array([], dtype=int)

    distances = phyl_dist.dist_matrix[org_index, valid_kegg]
    min_dist = float(distances.min())
    keep = [
        valid_rows[i] for i in range(len(valid_rows))
        if distances[i] == min_dist
    ]
    return np.asarray(keep, dtype=int)
