"""Parse a proteomics.tsv into a ProtData with optional filtering.

Ported from GECKO MATLAB:
src/io/loadProtData.m.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from .pax_db_loader import ProtData


_NAN_TOKENS = {"NA", "na", "NaN", "#VALUE!", ""}


def load_prot_data(
    source: Union[str, Path, ProtData],
    repl_per_cond: list[int],
    *,
    filter_data: bool = True,
    min_val: float = 0.0,
    max_rsd: float = 1.0,
    max_missing: Union[float, list[float]] = 2.0 / 3.0,
    cut_lowest: float = 5.0,
    add_stdevs: float = 1.0,
    zeros_as_measured: bool = False,
) -> ProtData:
    """Load (and optionally filter) absolute proteomics data.

    The input file is tab-delimited with one header row and columns
    ``uniprot, repl1, repl2, ..., replN`` where the replicate columns
    are concatenated across conditions in the order given by
    ``repl_per_cond`` (e.g. ``[3, 2]`` -> first three columns are
    condition 0's triplicate, next two are condition 1's duplicate).
    Levels are in mg/gDCW.

    An abundance of 0 is read as not measured (NaN) before any filter,
    unless ``zeros_as_measured`` is set. This applies to a file and to a
    ``ProtData`` source, with ``filter_data`` True and False.

    For each condition, the per-replicate measurements are reduced
    to a single value per protein:

    1. ``filter_data=True`` applies the following pipeline (matching
       MATLAB):

       * **maxMissing:** rows with fewer than
         ``max_missing * n_replicates`` strictly-positive
         measurements (with ``zeros_as_measured``, measurements of 0
         or above) have all replicates set to NaN.
       * **RSD:** rows whose ``std/mean > max_rsd`` are set to NaN.
       * **Collapse:** per-row ``mean + add_stdevs * std``.
       * **minVal:** collapsed values below ``min_val`` -> NaN (with
         ``zeros_as_measured``, a collapsed value of 0 is kept).
       * **Bottom cut:** among non-NaN values, the lowest
         ``cut_lowest`` percent are set to NaN (with
         ``zeros_as_measured``, the values of 0 are left out of the
         count and of the cut).

    2. ``filter_data=False`` just takes the per-row mean (NaN-safe).

    Rows that end up NaN in every condition are dropped, along with
    their UniProt IDs. A protein that is not measured in a condition is
    NaN in ``abundances`` and leaves the usage of its enzyme drawing from
    the protein pool (see ``fill_enz_concs`` and ``constrain_enz_concs``).

    Ported from GECKO MATLAB:
    src/io/loadProtData.m.

    Parameters
    ----------
    source
        Path to the TSV file, OR a pre-loaded ``ProtData`` to
        re-filter (matching MATLAB).
    repl_per_cond
        Number of replicates per condition. The total of these must
        equal the number of replicate columns in the file.
    filter_data
        Whether to apply the filtering pipeline. When False, every
        other parameter except ``zeros_as_measured`` is unused.
    min_val
        Minimum collapsed value to keep.
    max_rsd
        Maximum relative standard deviation per row.
    max_missing
        Minimum fraction of strictly-positive replicates required
        (replicates of 0 or above with ``zeros_as_measured``).
        May be a single float (applied to all conditions) or a list
        of length ``len(repl_per_cond)``.
    cut_lowest
        Percentage of lowest collapsed values to discard per
        condition.
    add_stdevs
        Number of standard deviations to add to the per-row mean
        when collapsing.
    zeros_as_measured
        Whether an abundance of 0 is a measurement. If False, a 0 is read
        as not measured (NaN) before any filter, also in a ``ProtData``
        source: the mean of a protein over its replicates leaves out the
        zeros, and a protein with only zeros is removed, so it draws from
        the protein pool. If True, a 0 counts as a measured replicate and
        is kept in the mean; a protein whose replicates are all 0 is
        returned as 0, is not removed by ``min_val`` or ``cut_lowest``, and
        limits the usage of its enzyme to 0 in ``constrain_enz_concs``.
        ``max_rsd`` and ``max_missing`` still apply to a protein with some
        zero replicates. Proteomics reports a protein below the detection
        limit as 0 or as missing. With True, a 0 is read as a protein that
        is absent, for example in a deletion strain. Applies with
        ``filter_data`` True and False.

    Returns
    -------
    ProtData
        ``abundances`` shape ``(n_proteins_kept, n_conditions)``.

    Raises
    ------
    FileNotFoundError
        If a path is given that does not exist.
    ValueError
        If ``repl_per_cond`` is empty, the file has no replicate
        columns, or the column count does not match.
    """
    if not repl_per_cond:
        raise ValueError("repl_per_cond must be non-empty.")
    total_replicates = sum(repl_per_cond)

    uniprot_ids, abundances = _load_raw(source, total_replicates)
    if not zeros_as_measured:
        abundances[abundances == 0] = np.nan

    # Drop empty UniProt IDs.
    keep = [bool(uid) for uid in uniprot_ids]
    uniprot_ids = [uid for uid, k in zip(uniprot_ids, keep) if k]
    abundances = abundances[keep, :]

    n_proteins = abundances.shape[0]
    n_conditions = len(repl_per_cond)
    filt_abund = np.full((n_proteins, n_conditions), np.nan, dtype=float)

    col_offset = 0
    for i, n_repl in enumerate(repl_per_cond):
        cond_block = abundances[:, col_offset : col_offset + n_repl].copy()
        col_offset += n_repl

        if filter_data:
            collapsed = _filter_condition(
                cond_block,
                min_val=min_val,
                max_rsd=max_rsd,
                max_missing=_max_missing_for(max_missing, i),
                cut_lowest=cut_lowest,
                add_stdevs=add_stdevs,
                zeros_as_measured=zeros_as_measured,
            )
        else:
            collapsed = _nanmean_axis1(cond_block)

        filt_abund[:, i] = collapsed

    not_all_nan = ~np.all(np.isnan(filt_abund), axis=1)
    return ProtData(
        uniprot_ids=[uid for uid, k in zip(uniprot_ids, not_all_nan) if k],
        abundances=filt_abund[not_all_nan, :],
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_raw(
    source: Union[str, Path, ProtData], total_replicates: int,
) -> tuple[list[str], np.ndarray]:
    """Read uniprot_ids and per-replicate abundances from `source`."""
    if isinstance(source, ProtData):
        if source.abundances.ndim != 2:
            raise ValueError(
                "Re-filtering requires source.abundances to be 2-D."
            )
        return list(source.uniprot_ids), np.array(source.abundances, dtype=float)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"proteomics file not found: {path}")

    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(line.rstrip("\n").split("\t"))

    if len(rows) < 1:
        raise ValueError(f"{path} is empty.")

    header = rows[0]
    data = rows[1:]
    if len(header) < 2:
        raise ValueError(
            f"{path}: header has only {len(header)} column(s); "
            f"expected at least 'uniprot' + one replicate column."
        )

    expected_cols = total_replicates + 1
    if len(header) != expected_cols:
        raise ValueError(
            f"{path}: header has {len(header)} column(s) but "
            f"repl_per_cond sums to {total_replicates}, "
            f"expecting {expected_cols} columns."
        )

    uniprot_ids: list[str] = []
    values: list[list[float]] = []
    for r in data:
        # Pad short rows so indexing is safe.
        while len(r) < expected_cols:
            r.append("")
        uniprot_ids.append(r[0].strip())
        values.append([_parse_float(c) for c in r[1:expected_cols]])

    if not values:
        return uniprot_ids, np.empty((0, total_replicates), dtype=float)

    return uniprot_ids, np.array(values, dtype=float)


def _parse_float(s: str) -> float:
    s = s.strip()
    if s in _NAN_TOKENS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _max_missing_for(
    max_missing: Union[float, list[float]], cond_idx: int,
) -> float:
    if isinstance(max_missing, (list, tuple, np.ndarray)):
        return float(max_missing[cond_idx])
    return float(max_missing)


def _nanmean_axis1(arr: np.ndarray) -> np.ndarray:
    """Per-row mean ignoring NaN; rows that are all-NaN return NaN
    (the all-NaN-slice warning is suppressed)."""
    if arr.size == 0:
        return np.empty(0, dtype=float)
    with warnings.catch_warnings(), np.errstate(
        invalid="ignore", divide="ignore",
    ):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=1)


def _nanstd_axis1(arr: np.ndarray) -> np.ndarray:
    """Per-row std ignoring NaN with ddof=1 (matching MATLAB's
    default ``std`` normalisation by N-1). Rows with <=1 valid
    sample return NaN; the ``Degrees of freedom <= 0`` warning is
    suppressed."""
    if arr.size == 0:
        return np.empty(0, dtype=float)
    with warnings.catch_warnings(), np.errstate(
        invalid="ignore", divide="ignore",
    ):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(arr, axis=1, ddof=1)


def _filter_condition(
    cond: np.ndarray,
    *,
    min_val: float,
    max_rsd: float,
    max_missing: float,
    cut_lowest: float,
    add_stdevs: float,
    zeros_as_measured: bool,
) -> np.ndarray:
    """Run the MATLAB filtering pipeline on one condition's
    (n_proteins, n_replicates) block. Returns a 1-D collapsed
    (n_proteins,) array."""
    cond = cond.copy()
    n_repl = cond.shape[1]

    # 1) maxMissing -- only when there's more than one replicate.
    if n_repl > 1:
        present = cond >= 0 if zeros_as_measured else cond > 0
        positive_count = np.sum(present, axis=1)
        threshold = max_missing * n_repl
        too_few = positive_count < threshold
        cond[too_few, :] = np.nan

    # 2) RSD filter.
    means = _nanmean_axis1(cond)
    stds = _nanstd_axis1(cond)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsd = np.where(means != 0, stds / means, np.nan)
    too_variable = rsd > max_rsd
    cond[too_variable, :] = np.nan

    # 3) Collapse: mean + add_stdevs * std.
    means = _nanmean_axis1(cond)
    stds = _nanstd_axis1(cond)
    # std is NaN for rows with <=1 sample; treat as 0 in the addition.
    extra = np.where(np.isnan(stds), 0.0, add_stdevs * stds)
    collapsed = means + extra

    # 4) minVal filter.
    too_small = collapsed < min_val
    if zeros_as_measured:
        too_small &= collapsed != 0
    collapsed[too_small] = np.nan

    # 5) Bottom-cut_lowest% filter.
    valid = ~np.isnan(collapsed)
    if zeros_as_measured:
        valid &= collapsed != 0
    n_valid = int(valid.sum())
    if n_valid > 0 and cut_lowest > 0:
        cut_n = int(math.floor(n_valid * cut_lowest / 100.0))
        if cut_n > 0:
            valid_indices = np.where(valid)[0]
            order = valid_indices[np.argsort(collapsed[valid_indices], kind="stable")]
            collapsed[order[:cut_n]] = np.nan

    return collapsed
