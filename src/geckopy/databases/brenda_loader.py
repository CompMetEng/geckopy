"""Parse BRENDA TSV dumps into kcat and SA-derived-kcat tables.

Ported from GECKO MATLAB:
src/geckomat/get_enzyme_data/loadBRENDAdata.m.

Three tab-delimited files are read from the BRENDA folder:

    kcat.tsv   kcat values (wide: one row per triple, both max and
               median across the raw measurements that fed into it)
    sa.tsv     specific activities (same wide shape)
    mw.tsv     molecular weights (single value per (ec, organism))

The files are produced by the ``geckopy brenda-refresh`` CLI. geckopy's
own come from a toolbox-assets release (see
:func:`~geckopy.databases.fetch_brenda_asset`). This loader reads plain
text, or ``.xz``-compressed copies (``kcat.tsv.xz`` etc.) when present.

kcat and SA have seven tab-delimited columns: EC number, substrate
(``*`` for SA), organism, value-max, value-median, n (number of raw
measurements aggregated), references (semicolon-joined PMIDs or
``*``). MW has six columns (single value, no aggregation choice). A
``#`` header line carries the BRENDA release version; the line after
it is the TSV column header (also skipped).
"""
from __future__ import annotations

import logging
import lzma
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_KCAT_COLUMNS = ["ec_code", "substrate", "organism", "kcat", "n"]
_SA_COLUMNS = ["ec_code", "organism", "kcat", "mw", "n"]


@dataclass
class BrendaData:
    """Loaded BRENDA tables.

    Each kcat / SA table is exposed as separate ``_max`` and ``_median``
    DataFrames built once at load time. Both views share the same row
    set (one per (ec, substrate, organism) triple); they differ only in
    which aggregation of the raw measurements fills the ``kcat``
    column. Choose between views via :meth:`kcat_for` / :meth:`sa_for`.

    Attributes
    ----------
    kcat_max, kcat_median
        DataFrames with columns ``ec_code`` (str), ``substrate`` (str),
        ``organism`` (str), ``kcat`` (float, 1/s), ``n`` (Int64).
    sa_max, sa_median
        Same shape, joined with MW to give an SA-derived kcat. Columns
        ``ec_code``, ``organism``, ``kcat`` (float, 1/s), ``mw`` (float,
        g/mmol), ``n`` (Int64).
    """
    kcat_max: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_KCAT_COLUMNS)
    )
    kcat_median: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_KCAT_COLUMNS)
    )
    sa_max: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_SA_COLUMNS)
    )
    sa_median: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_SA_COLUMNS)
    )

    def kcat_for(self, aggregation: Literal["max", "median"]) -> pd.DataFrame:
        """Return the kcat view for the requested aggregation."""
        if aggregation == "median":
            return self.kcat_median
        if aggregation == "max":
            return self.kcat_max
        raise ValueError(
            f"aggregation must be 'max' or 'median', got {aggregation!r}"
        )

    def sa_for(self, aggregation: Literal["max", "median"]) -> pd.DataFrame:
        """Return the SA-derived-kcat view for the requested aggregation."""
        if aggregation == "median":
            return self.sa_median
        if aggregation == "max":
            return self.sa_max
        raise ValueError(
            f"aggregation must be 'max' or 'median', got {aggregation!r}"
        )


def load_brenda_data(
    folder: str | Path, *, filter_outlier_kcats: bool = True,
) -> BrendaData:
    """Load BRENDA kcat, SA, and MW dumps from a folder.

    Ported from GECKO MATLAB:
    src/geckomat/get_enzyme_data/loadBRENDAdata.m.

    Reads ``kcat.tsv``, ``sa.tsv``, ``mw.tsv`` from ``folder``, splits
    the kcat and SA wide tables into per-aggregation views, applies the
    GECKO unit conversions, and joins SA + MW on EC + organism
    (case-insensitive) to produce derived kcat tables.

    Unit conventions:

    * KCAT: file values are in 1/s already; no scaling.
    * SA:   file values in ``[umol/min/mg]`` are multiplied by ``1/60``
            to give ``[mmol/s/g]``.
    * MW:   file value in ``[g/mol]`` is multiplied by ``1/1000`` to
            give ``[g/mmol]``.
    * Derived kcat in the SA table is ``SA * MW = [1/s]``.

    Both kcat and SA are loaded as separate max and median views over
    the raw measurements aggregated per (ec, substrate, organism)
    triple, so a project can opt into median aggregation via
    ``adapter.params.kcat_aggregate_brenda`` without regenerating the
    snapshot. See ``docs/kcat_aggregation.md`` for the rationale.

    Malformed lines (wrong number of fields, non-numeric value column)
    are skipped with a ``logger.warning``; the rest of the file is
    parsed normally.

    Parameters
    ----------
    folder
        Directory containing all three BRENDA dump files.
    filter_outlier_kcats
        If True (default), drop a row whose kcat is more than 4 orders
        of magnitude below the typical kcat of its EC before it can be
        matched to any reaction -- see ``_filter_outlier_rows`` for the
        rationale and mechanism. Set False to reproduce the unfiltered
        1.0-era behaviour.

    Returns
    -------
    BrendaData
        Both kcat views may be empty if the source files were empty or
        contained only malformed lines; both SA views may be empty if
        no SA row matched any MW row.

    Raises
    ------
    FileNotFoundError
        If any of the three expected files is missing.
    """
    folder = Path(folder)
    kcat_path = _resolve_brenda_path(folder, "kcat")
    sa_path = _resolve_brenda_path(folder, "sa")
    mw_path = _resolve_brenda_path(folder, "mw")

    for p in (kcat_path, sa_path, mw_path):
        if not p.is_file():
            raise FileNotFoundError(f"Expected BRENDA file not found: {p}")

    kcat_wide = _load_wide_table(
        kcat_path, value_columns=("kcat_max", "kcat_median"),
        value_scale=1.0,
    )
    sa_wide = _load_wide_table(
        sa_path, value_columns=("sa_max", "sa_median"),
        value_scale=1.0 / 60.0,
    )
    mw_raw = _load_mw_table(mw_path, value_scale=1.0 / 1000.0)

    kcat_max = _view_from_wide(kcat_wide, "kcat_max", _KCAT_COLUMNS)
    kcat_median = _view_from_wide(kcat_wide, "kcat_median", _KCAT_COLUMNS)
    sa_max = _join_sa_with_mw(
        _sa_view_from_wide(sa_wide, "sa_max"), mw_raw,
    )
    sa_median = _join_sa_with_mw(
        _sa_view_from_wide(sa_wide, "sa_median"), mw_raw,
    )

    if filter_outlier_kcats:
        kcat_max, kcat_median, sa_max, sa_median = _filter_outlier_rows(
            [kcat_max, kcat_median, sa_max, sa_median],
        )

    return BrendaData(
        kcat_max=kcat_max,
        kcat_median=kcat_median,
        sa_max=sa_max,
        sa_median=sa_median,
    )


# --------------------------------------------------------------------------- #
# File parsing
# --------------------------------------------------------------------------- #


def _resolve_brenda_path(folder: Path, stem: str) -> Path:
    """``<stem>.tsv.xz`` when present, otherwise ``<stem>.tsv``."""
    xz_path = folder / f"{stem}.tsv.xz"
    if xz_path.is_file():
        return xz_path
    return folder / f"{stem}.tsv"


def _open_brenda_file(path: Path) -> IO[str]:
    """Open a BRENDA TSV for text reading, transparently decompressing
    ``.xz`` files."""
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# Wide-format file layout (kcat / SA): 7 columns
# 0=ec_code, 1=substrate, 2=organism, 3=value_max, 4=value_median,
# 5=n, 6=refs
_WIDE_NCOLS = 7
# MW file layout: 6 columns
# 0=ec_code, 1=substrate, 2=organism, 3=value, 4=n, 5=refs
_MW_NCOLS = 6


def _load_wide_table(
    path: Path,
    *,
    value_columns: tuple[str, str],
    value_scale: float,
) -> pd.DataFrame:
    """Parse a 7-column kcat/SA TSV (wide format) into a DataFrame.

    Returns a DataFrame with columns ``ec_code``, ``substrate``,
    ``organism``, ``<value_columns[0]>``, ``<value_columns[1]>``, ``n``.
    The references column is dropped. Comment lines (``#``-prefixed)
    and the column-header row are skipped.
    """
    max_col, med_col = value_columns
    rows: list[tuple[str, str, str, float, float, int]] = []
    invalid = 0
    with _open_brenda_file(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Skip the TSV column-header row (string in a numeric column).
            if line_no <= 3 and parts[0] == "ec_code":
                continue
            if len(parts) < _WIDE_NCOLS:
                invalid += 1
                logger.warning(
                    "%s:%d: expected %d tab-delimited fields, got %d; skipping.",
                    path.name, line_no, _WIDE_NCOLS, len(parts),
                )
                continue
            ec, substrate, organism, vmax_s, vmed_s, n_s, _ = parts[:_WIDE_NCOLS]
            try:
                vmax = float(vmax_s) * value_scale
                vmed = float(vmed_s) * value_scale
                n = int(n_s)
            except ValueError:
                invalid += 1
                logger.warning(
                    "%s:%d: non-numeric value(s) (%r, %r, %r); skipping.",
                    path.name, line_no, vmax_s, vmed_s, n_s,
                )
                continue
            rows.append((ec, substrate, organism, vmax, vmed, n))

    if invalid:
        logger.warning("%s: skipped %d malformed line(s).", path.name, invalid)

    return pd.DataFrame(
        rows,
        columns=["ec_code", "substrate", "organism", max_col, med_col, "n"],
    )


def _load_mw_table(path: Path, *, value_scale: float) -> pd.DataFrame:
    """Parse the 6-column MW TSV (no aggregation choice).

    Returns a DataFrame with columns ``ec_code``, ``organism``,
    ``value``. ``substrate`` is always ``*`` and is dropped on the way
    out; ``n`` and the references column are dropped too.
    """
    rows: list[tuple[str, str, float]] = []
    invalid = 0
    with _open_brenda_file(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if line_no <= 3 and parts[0] == "ec_code":
                continue
            if len(parts) < _MW_NCOLS:
                invalid += 1
                logger.warning(
                    "%s:%d: expected %d tab-delimited fields, got %d; skipping.",
                    path.name, line_no, _MW_NCOLS, len(parts),
                )
                continue
            ec, _, organism, value_str, _, _ = parts[:_MW_NCOLS]
            try:
                value = float(value_str) * value_scale
            except ValueError:
                invalid += 1
                logger.warning(
                    "%s:%d: column 4 %r is not numeric; skipping.",
                    path.name, line_no, value_str,
                )
                continue
            rows.append((ec, organism, value))

    if invalid:
        logger.warning("%s: skipped %d malformed line(s).", path.name, invalid)

    return pd.DataFrame(rows, columns=["ec_code", "organism", "value"])


def _view_from_wide(
    wide: pd.DataFrame, value_col: str, out_columns: list[str],
) -> pd.DataFrame:
    """Project the wide kcat table down to one of its value columns.

    Returns a DataFrame with the public schema
    ``[ec_code, substrate, organism, kcat, n]``.
    """
    if wide.empty:
        return pd.DataFrame(columns=out_columns)
    sub = wide[["ec_code", "substrate", "organism", value_col, "n"]].copy()
    sub = sub.rename(columns={value_col: "kcat"})
    return sub.reset_index(drop=True)


def _sa_view_from_wide(wide: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Project the wide SA table to one aggregation, dropping substrate.

    Returns a DataFrame with columns ``ec_code``, ``organism``, ``sa``,
    ``n``. The MW join helper attaches MW + computes ``kcat = sa * mw``.
    """
    if wide.empty:
        return pd.DataFrame(columns=["ec_code", "organism", "sa", "n"])
    sub = wide[["ec_code", "organism", value_col, "n"]].copy()
    sub = sub.rename(columns={value_col: "sa"})
    return sub.reset_index(drop=True)


def _join_sa_with_mw(sa: pd.DataFrame, mw: pd.DataFrame) -> pd.DataFrame:
    """Join SA + MW on (EC, organism) case-insensitive, first MW match wins.

    For each SA row, find the first MW row with the same EC and
    organism (case-insensitive). If found, emit a row with the
    SA-derived kcat. SA rows without a matching MW row are dropped,
    matching MATLAB.
    """
    if sa.empty or mw.empty:
        return pd.DataFrame(columns=_SA_COLUMNS)

    mw_lookup: dict[tuple[str, str], float] = {}
    for ec, organism, mw_value in zip(
        mw["ec_code"].str.upper(),
        mw["organism"].str.upper(),
        mw["value"],
    ):
        mw_lookup.setdefault((ec, organism), mw_value)

    out_rows: list[dict[str, object]] = []
    for ec, organism, sa_value, n in zip(
        sa["ec_code"], sa["organism"], sa["sa"], sa["n"],
    ):
        mw_value = mw_lookup.get((ec.upper(), organism.upper()))
        if mw_value is None:
            continue
        out_rows.append({
            "ec_code": ec,
            "organism": organism,
            "kcat": sa_value * mw_value,
            "mw": mw_value,
            "n": int(n),
        })

    if not out_rows:
        return pd.DataFrame(columns=_SA_COLUMNS)
    return pd.DataFrame(out_rows, columns=_SA_COLUMNS)


def _filter_outlier_rows(
    tables: list[pd.DataFrame], *, decades: float = 4.0, min_ref: int = 5,
) -> list[pd.DataFrame]:
    """Drop a row whose kcat is implausibly low for its EC.

    BRENDA's specific-activity x molecular-weight join can produce a
    derived kcat that is many orders of magnitude below anything else
    reported for the same EC -- almost always a unit or transcription
    error upstream, not a real slow enzyme (a documented case: EC
    2.3.1.297's only molecular-weight row is for a different organism
    than its lowest specific-activity row, and the derived kcat this
    produces, ~4e-7 1/s, is 8 orders of magnitude below the yeast SA
    measurement for the same EC that has no matching MW row and is
    silently dropped instead). Once matched to a reaction, a kcat that
    small makes the enzyme-usage coefficient enormous and can make a
    model untunable.

    This is a *relative* floor, not a fixed threshold: a row is dropped
    only if it is more than ``decades`` orders of magnitude below the
    median kcat of its own EC. "Its own EC" widens from the full
    4-level code to the 3-, 2-, then 1-level prefix if fewer than
    ``min_ref`` values are available at the current level, so a
    sparsely-represented EC is still judged against something. The
    reference pool for one EC is built from every positive-kcat row of
    every table passed in (plain kcat rows and SA-derived rows alike),
    computed once up front from the *unfiltered* tables so that
    dropping a row in one table cannot shift the reference used to
    judge another.

    Real enzymes can be genuinely slow, so this only catches a row far
    outside its own EC's spread, not a merely-low one.

    Parameters
    ----------
    tables
        The four BRENDA views (``kcat_max``, ``kcat_median``,
        ``sa_max``, ``sa_median``); each must have ``ec_code`` and
        ``kcat`` columns. Order is preserved in the return value.
    decades
        How many orders of magnitude below the reference counts as an
        outlier.
    min_ref
        Minimum number of reference values required before a level is
        trusted; the search widens to a broader EC prefix otherwise.

    Returns
    -------
    list[pandas.DataFrame]
        The same tables, each with outlier rows removed and its index
        reset.
    """
    pool: dict[str, list[float]] = {}
    for table in tables:
        for ec, kcat in zip(table["ec_code"], table["kcat"]):
            if kcat > 0 and "-" not in ec:
                pool.setdefault(ec, []).append(np.log10(kcat))

    log_values_by_level: dict[tuple[int, str], list[float]] = {}
    for ec, values in pool.items():
        parts = ec.split(".")
        for level in (4, 3, 2, 1):
            key = (level, ".".join(parts[:level]))
            log_values_by_level.setdefault(key, []).extend(values)

    reference_cache: dict[str, float | None] = {}

    def reference_log_kcat(ec: str) -> float | None:
        if ec not in reference_cache:
            parts = ec.split(".")
            reference_cache[ec] = None
            for level in (4, 3, 2, 1):
                values = log_values_by_level.get((level, ".".join(parts[:level])), [])
                if len(values) >= min_ref:
                    reference_cache[ec] = float(np.median(values))
                    break
        return reference_cache[ec]

    filtered = []
    for table in tables:
        if table.empty:
            # An empty boolean mask loses the (already-correct) column
            # set on an empty frame; nothing to filter anyway.
            filtered.append(table)
            continue
        keep = []
        for ec, kcat in zip(table["ec_code"], table["kcat"]):
            ref = reference_log_kcat(ec) if kcat > 0 else None
            keep.append(not (ref is not None and np.log10(kcat) < ref - decades))
        filtered.append(table[pd.Series(keep, index=table.index)].reset_index(drop=True))
    return filtered
