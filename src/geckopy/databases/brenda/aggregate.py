"""Aggregate parsed BRENDA rows per (ec, substrate, organism) and write
the three snapshot TSVs.

For kcat and SA the file is **wide**: one row per (ec, substrate,
organism) triple holding both the max and median of all raw
measurements that fell into the triple, plus ``n`` (count of raw
measurements aggregated). MW is a per-protein physical property, not an
aggregation question — one row per (ec, organism) with a single ``mw``
column.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from .parse import Kind, Row

logger = logging.getLogger(__name__)


# Output filenames per kind. Plain ``<kind>.tsv``; the historical
# ``max_<kind>.tsv`` name would be wrong now that the same row has
# both max and median.
_OUTPUT_FILES: dict[Kind, tuple[str, str]] = {
    "kcat": ("kcat.tsv", "kcat in 1/s"),
    "sa":   ("sa.tsv",   "specific activity in umol/min/mg"),
    "mw":   ("mw.tsv",   "molecular weight in g/mol"),
}

_HEADER_FMT = (
    "# BRENDA release {release} generated {date} - CC BY 4.0 - {desc}\n"
)

# Wide column layouts per kind. Written as a TSV header row after the
# ``#`` release comment so downstream readers can validate the layout.
_KCAT_HEADERS = (
    "ec_code", "substrate", "organism",
    "kcat_max", "kcat_median", "n", "references",
)
_SA_HEADERS = (
    "ec_code", "substrate", "organism",
    "sa_max", "sa_median", "n", "references",
)
_MW_HEADERS = (
    "ec_code", "substrate", "organism", "mw", "n", "references",
)

_COLUMN_HEADERS: dict[Kind, tuple[str, ...]] = {
    "kcat": _KCAT_HEADERS,
    "sa":   _SA_HEADERS,
    "mw":   _MW_HEADERS,
}

# kcat and SA have both max and median per triple; MW is single-valued.
_DUAL_AGG_KINDS: tuple[Kind, ...] = ("kcat", "sa")


def aggregate_and_write(
    rows: Iterable[Row],
    out_dir: str | Path,
    *,
    release: str,
    date: str | None = None,
) -> dict[Kind, Path]:
    """Group rows by (kind, ec, substrate, organism) and write three TSVs.

    For kcat and SA the snapshot keeps one wide row per triple with
    both max and median columns; for MW only the per-triple max is
    written.

    Parameters
    ----------
    rows
        Iterable of ``Row`` from ``parse_brenda_json``.
    out_dir
        Output directory for ``kcat.tsv``, ``sa.tsv``, ``mw.tsv``.
        Created if missing.
    release
        BRENDA release string, e.g. ``"2026.1"``. Written into the
        header line of each TSV.
    date
        ISO date string for the header. Defaults to today.

    Returns
    -------
    dict
        Mapping of ``Kind`` to written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if date is None:
        date = dt.date.today().isoformat()

    # Bucket raw rows per (kind, ec, substrate, organism). For each
    # bucket we keep every value and the union of all PMID refs so we
    # can compute both max and median.
    buckets: dict[
        tuple[Kind, str, str, str], tuple[list[float], set[str]]
    ] = {}
    for r in rows:
        key = (r.kind, r.ec, r.substrate, r.organism)
        entry = buckets.get(key)
        if entry is None:
            buckets[key] = ([r.value], set(r.references))
        else:
            entry[0].append(r.value)
            entry[1].update(r.references)

    paths: dict[Kind, Path] = {}
    for kind, (basename, desc) in _OUTPUT_FILES.items():
        path = out_dir / basename
        paths[kind] = path
        keys = sorted(
            (k for k in buckets if k[0] == kind),
            key=lambda k: (k[1], k[2], k[3]),
        )
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(_HEADER_FMT.format(release=release, date=date, desc=desc))
            fh.write("\t".join(_COLUMN_HEADERS[kind]) + "\n")
            for k in keys:
                values, refs = buckets[k]
                refs_field = ";".join(sorted(refs)) if refs else "*"
                vmax = float(np.max(values))
                _, ec, sub, org = k
                if kind in _DUAL_AGG_KINDS:
                    vmed = float(np.median(values))
                    fh.write(
                        f"{ec}\t{sub}\t{org}\t"
                        f"{_fmt_value(vmax)}\t{_fmt_value(vmed)}\t"
                        f"{len(values)}\t{refs_field}\n"
                    )
                else:
                    fh.write(
                        f"{ec}\t{sub}\t{org}\t{_fmt_value(vmax)}\t"
                        f"{len(values)}\t{refs_field}\n"
                    )
        logger.info("wrote %d rows to %s", len(keys), path)
    return paths


def _fmt_value(v: float) -> str:
    # Python ``str(float)`` always includes a decimal point or exponent,
    # so ``598.0`` and ``2.5`` are written and read back identically.
    return str(v)
