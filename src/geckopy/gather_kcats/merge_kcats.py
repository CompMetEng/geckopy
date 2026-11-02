"""Merge kcat lists from multiple sources by tiered priority.

Generalizes GECKO MATLAB's ``mergeDLKcatAndFuzzyKcats``: rather than two
fixed inputs (DLKcat + fuzzy BRENDA), :func:`merge_kcats` accepts any
number of kcat lists, each of which may itself mix several sources. An
OpenKineticsPredictor result, for example, is a single list whose rows
carry ``BRENDA``, ``Sabio-RK``, ``CataPro`` or ``UniProt`` in their
``source`` column. A caller-supplied ``source_priority`` decides which
source wins for each reaction.

Three generic tier tokens may appear in ``source_priority`` instead of a
literal source name:

- ``"database_exact"`` -- an exact experimental-database measurement: an
  experimental-database row (``BRENDA`` / ``Sabio-RK``) that carries no
  fuzzy metadata, i.e. a direct OKP hit with no EC wildcards. These are
  the most precise values and are usually ranked first.
- ``"database_top"`` -- a high-confidence *fuzzy* BRENDA match with
  ``wildcard_level == 0`` and ``origin <= top_origin_limit``.
- ``"database_bottom"`` -- weaker fuzzy BRENDA matches still within
  ``wildcard_limit`` / ``bottom_origin_limit``.

Anything else in ``source_priority`` is matched against the row's
``source`` after normalisation to lowercase ``snake_case`` (so
``"DLKcat"`` matches ``"dlkcat"`` and ``"Sabio-RK"`` matches
``"sabio_rk"``). A row whose source carries the ``OKP-`` pipeline-stage
tag (``parse_okp_output``'s output, e.g. ``"OKP-BRENDA"``) is matched
and tiered by its underlying source with the tag stripped -- an
``OKP-BRENDA`` row still routes to the database tiers like a bare
``BRENDA`` row would -- while the merged output keeps the original,
tagged ``source`` string.

Ported from / generalizes GECKO MATLAB:
src/geckomat/gather_kcats/mergeDLKcatAndFuzzyKcats.m.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = [
    "rxn_id", "source", "eccode", "substrates", "genes",
    "kcat", "wildcard_level", "origin",
]

#: Generic tier tokens recognised in ``source_priority``. They select
#: experimental-database rows by quality rather than by source label.
DATABASE_EXACT = "database_exact"
DATABASE_TOP = "database_top"
DATABASE_BOTTOM = "database_bottom"

#: Source labels (normalised) treated as experimental databases, i.e.
#: routed to the ``database_top`` / ``database_bottom`` tiers.
_DEFAULT_DATABASE_SOURCES = ("brenda", "sabio_rk")

#: Normalised prefix ``parse_okp_output`` tags a row's source with
#: (``"OKP-BRENDA"`` normalises to ``"okp_brenda"``) to mark it as
#: having come through the OpenKineticsPredictor pipeline stage.
_OKP_PREFIX = "okp_"


def normalize_source(label: object) -> str:
    """Fold a source label to lowercase ``snake_case``.

    ``"Sabio-RK"`` -> ``"sabio_rk"``, ``"DLKcat"`` -> ``"dlkcat"``,
    ``"CataPro"`` -> ``"catapro"``. Runs of non-alphanumeric characters
    collapse to a single underscore; leading/trailing underscores are
    stripped.
    """
    s = re.sub(r"[^0-9a-z]+", "_", str(label).strip().lower())
    return s.strip("_")


def _strip_okp_prefix(normalized_source: str) -> str:
    """Strip a leading ``okp_`` from an already-normalised source.

    Tiering and ``source_priority`` matching key off the underlying
    source (``"okp_brenda"`` must still route to the database tiers
    like ``"brenda"``; ``"okp_catapro"`` must still match a
    ``source_priority`` entry for ``"catapro"``); only the row's own
    ``source`` column, untouched by this function, keeps the ``OKP-``
    tag through to the merged output.
    """
    if normalized_source.startswith(_OKP_PREFIX):
        return normalized_source[len(_OKP_PREFIX):]
    return normalized_source


def merge_kcats(
    *kcat_lists: pd.DataFrame,
    source_priority: Sequence[str],
    top_origin_limit: int = 4,
    bottom_origin_limit: int = 6,
    wildcard_limit: int = 3,
    database_sources: Iterable[str] = _DEFAULT_DATABASE_SOURCES,
) -> pd.DataFrame:
    """Merge any number of kcat lists, keeping the best source per reaction.

    All input lists are concatenated and each row is assigned a tier
    (see the module docstring). For every reaction the highest-priority
    tier present wins, and **all** rows of that tier for that reaction
    are kept; rows of lower-priority tiers for the same reaction are
    dropped.

    Rows whose ``kcat`` is non-positive or missing carry no usable
    turnover number and are dropped first. This also removes the
    ``kcat == 0`` / ``origin == NA`` rows that ``fuzzy_kcat_matching``
    emits for unmatched reactions, so a leftover NA-metadata
    experimental-database row is unambiguously an exact OKP value and
    is routed to ``database_top``.

    Parameters
    ----------
    *kcat_lists
        One or more DataFrames in the canonical 8-column kcat-list
        schema. A single list may mix several sources.
    source_priority
        Ordered sequence (best first) of tier tokens / source labels.
        Tokens are normalised the same way as row sources, except the
        reserved ``"database_top"`` / ``"database_bottom"`` tiers. A
        source not listed here is dropped (with a warning).
    top_origin_limit
        Origin upper bound for ``database_top`` fuzzy rows. ``[1, 6]``.
        Default ``4``, not ``6``: origins 5-6 are specific-activity ×
        molecular-weight derived kcats, which BRENDA's own data quality
        does not always support treating as better than a prediction
        (a bad SA/MW join can otherwise silently override a good
        prediction with an outlier -- see ``load_brenda_data``'s
        ``filter_outlier_kcats``, which removes the worst of these
        before they reach this function, but does not catch every one).
    bottom_origin_limit
        Origin upper bound for ``database_bottom`` fuzzy rows. ``[1, 6]``.
    wildcard_limit
        Maximum wildcard count for the ``database_bottom`` wildcard
        branch. ``[0, 3]``.
    database_sources
        Source labels treated as experimental databases (routed to the
        ``database_*`` tiers). Defaults to ``("brenda", "sabio_rk")``.

    Returns
    -------
    pandas.DataFrame
        Merged kcat list in the canonical 8-column schema. Row order
        follows the input lists (each list's surviving rows, in turn),
        matching MATLAB ``mergeKcats``. The original (un-normalised)
        ``source`` strings are preserved.

    Raises
    ------
    ValueError
        If no list is given, ``source_priority`` is empty, a limit is
        out of range, or any input list is missing required columns.
    """
    if not kcat_lists:
        raise ValueError("merge_kcats requires at least one kcat list.")
    if not (1 <= top_origin_limit <= 6):
        raise ValueError(
            f"top_origin_limit must be in [1, 6], got {top_origin_limit}"
        )
    if not (1 <= bottom_origin_limit <= 6):
        raise ValueError(
            f"bottom_origin_limit must be in [1, 6], got {bottom_origin_limit}"
        )
    if not (0 <= wildcard_limit <= 3):
        raise ValueError(
            f"wildcard_limit must be in [0, 3], got {wildcard_limit}"
        )
    if not list(source_priority):
        raise ValueError("source_priority must be a non-empty sequence.")

    for i, df in enumerate(kcat_lists):
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"kcat_lists[{i}] missing required column(s): {missing}"
            )

    db_sources = {normalize_source(s) for s in database_sources}
    rank: dict[str, int] = {}
    for r, token in enumerate(source_priority):
        rank.setdefault(normalize_source(token), r)

    parts = [df[_REQUIRED_COLUMNS] for df in kcat_lists if not df.empty]
    if not parts:
        return _empty()
    combined = (
        pd.concat(parts, ignore_index=True)
        if len(parts) > 1
        else parts[0].reset_index(drop=True)
    )

    # Drop rows without a usable turnover number (also removes the
    # kcat == 0 unmatched rows fuzzy_kcat_matching emits).
    kcat_num = pd.to_numeric(combined["kcat"], errors="coerce")
    valid = kcat_num > 0
    dropped = int((~valid).sum())
    if dropped:
        logger.debug(
            "merge_kcats: dropped %d row(s) with non-positive/missing kcat.",
            dropped,
        )
    combined = combined[valid]
    if combined.empty:
        return _empty()

    norm = combined["source"].map(normalize_source)
    match_key = norm.map(_strip_okp_prefix)
    wc = pd.to_numeric(combined["wildcard_level"], errors="coerce")
    origin = pd.to_numeric(combined["origin"], errors="coerce")
    is_db = match_key.isin(db_sources)
    meta = wc.notna() & origin.notna()

    top = (wc == 0) & (origin <= top_origin_limit)
    bottom = (
        ((wc == 0) & (origin > top_origin_limit) & (origin <= bottom_origin_limit))
        | ((wc > 0) & (wc <= wildcard_limit) & (origin <= bottom_origin_limit))
    )
    db_exact = is_db & (~meta)            # exact OKP DB hit (no wildcards)
    db_top = is_db & meta & top
    db_bottom = is_db & meta & (~db_top) & bottom

    # Non-database rows keep their (normalised, OKP-prefix-stripped)
    # source as the tier; the database rows are re-tiered, and those
    # passing neither gate stay NA and are dropped.
    tier = match_key.copy()
    tier[is_db] = pd.NA
    tier[db_exact] = DATABASE_EXACT
    tier[db_top] = DATABASE_TOP
    tier[db_bottom] = DATABASE_BOTTOM

    unknown = sorted(s for s in set(match_key[~is_db]) if s not in rank)
    if unknown:
        logger.warning(
            "merge_kcats: %d source(s) not listed in source_priority were "
            "dropped: %s", len(unknown), unknown,
        )

    rank_series = tier.map(rank)
    eligible = rank_series.notna()
    if not eligible.any():
        return _empty()

    work = combined[eligible].copy()
    work["__rank"] = rank_series[eligible]
    best = work.groupby("rxn_id")["__rank"].transform("min")
    # Keep input order (do not sort by tier) so the output matches MATLAB
    # mergeKcats; the boolean mask preserves the concatenation order.
    kept = work[work["__rank"] == best]
    return kept[_REQUIRED_COLUMNS].reset_index(drop=True)


def _empty() -> pd.DataFrame:
    """Return an empty kcat list with the canonical columns and dtypes."""
    df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    df["wildcard_level"] = df["wildcard_level"].astype("Int64")
    df["origin"] = df["origin"].astype("Int64")
    return df
