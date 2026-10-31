"""Merge DLKcat and fuzzy-BRENDA kcat lists by priority.

Thin backward-compatible wrapper around :func:`merge_kcats`, kept for the
common DLKcat + fuzzy-BRENDA case and for parity with the MATLAB name.
New code should call :func:`merge_kcats` directly, which also handles
single lists that mix several sources (e.g. an OpenKineticsPredictor
result with BRENDA / Sabio-RK / CataPro rows).

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/mergeDLKcatAndFuzzyKcats.m.
"""
from __future__ import annotations

import warnings

import pandas as pd

from .merge_kcats import DATABASE_BOTTOM, DATABASE_TOP, merge_kcats


def merge_dlkcat_and_fuzzy_kcats(
    dlkcat_kcats: pd.DataFrame,
    fuzzy_kcats: pd.DataFrame,
    *,
    top_origin_limit: int = 4,
    bottom_origin_limit: int = 6,
    wildcard_limit: int = 3,
) -> pd.DataFrame:
    """Merge DLKcat predictions with fuzzy BRENDA matches.

    Deprecated alias for :func:`merge_kcats` with the tiered priority
    ``[database_top, dlkcat, database_bottom]`` (BRENDA preferred over
    DLKcat, with weaker fuzzy matches as the final fallback). Will be
    removed in a future release; switch to ``merge_kcats``.

    Priority (highest first):

    1. **database_top:** fuzzy ``wildcard_level == 0`` AND
       ``origin <= top_origin_limit`` (the best BRENDA matches).
    2. **dlkcat:** any DLKcat row whose reaction is not covered by (1).
    3. **database_bottom:** weaker fuzzy matches
       (``wc == 0 AND top < origin <= bottom`` OR
       ``0 < wc <= wildcard_limit AND origin <= bottom``) whose reaction
       is not covered by (2).

    Parameters
    ----------
    dlkcat_kcats
        DataFrame produced by ``read_dlkcat_output``.
    fuzzy_kcats
        DataFrame produced by ``fuzzy_kcat_matching``.
    top_origin_limit
        Origin upper bound for database_top. Must be in ``[1, 6]``.
        Default 4.
    bottom_origin_limit
        Origin upper bound for database_bottom. Must be in ``[1, 6]``.
    wildcard_limit
        Maximum wildcard count for the database_bottom wildcard branch.
        Must be in ``[0, 3]``.

    Returns
    -------
    pandas.DataFrame
        Merged kcat list in the canonical 8-column schema.

    Raises
    ------
    ValueError
        If any limit is outside its allowed range, or if either input
        is missing required columns.
    """
    warnings.warn(
        "merge_dlkcat_and_fuzzy_kcats is deprecated; use merge_kcats "
        "instead. The old name will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return merge_kcats(
        fuzzy_kcats,
        dlkcat_kcats,
        source_priority=[DATABASE_TOP, "dlkcat", DATABASE_BOTTOM],
        top_origin_limit=top_origin_limit,
        bottom_origin_limit=bottom_origin_limit,
        wildcard_limit=wildcard_limit,
    )
