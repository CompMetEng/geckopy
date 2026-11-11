"""Parse a paxDB.tsv proteomics dump into a ProtData container.

Ported from GECKO MATLAB:
the inline paxDB-parsing block of
src/limit_proteins/calculateFfactor.m.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .uniprot_loader import UniprotDB


# Strips the leading `<digits>.` modifier on gene IDs (paxDB internal
# species/version prefix, e.g. "9606.ENSP00000123456").
_GENE_PREFIX_RE = re.compile(r"^\d+\.")


@dataclass
class ProtData:
    """Pre-loaded proteomics data for f-factor and concentration filling.

    Attributes
    ----------
    uniprot_ids
        UniProt IDs (one per protein measurement).
    abundances
        Numeric abundances. Shape ``(n_proteins,)`` for a single
        sample, or ``(n_proteins, n_samples)`` if multiple datasets
        were merged. Units are arbitrary (only ratios are used).
    """

    uniprot_ids: list[str] = field(default_factory=list)
    abundances: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


def load_pax_db(
    path: str | Path,
    uniprot_db: "UniprotDB",
) -> ProtData:
    """Parse a paxDB.tsv file and return a ``ProtData``.

    The file is tab-delimited with one or more ``#``-prefixed header
    lines and three data columns: ``internal_id``, ``gene_id``,
    ``level``. Gene IDs are stripped of any leading
    ``<digits>.`` modifier (paxDB's internal species/version prefix)
    before matching against ``uniprot_db.genes``. Rows whose gene
    name is not in the UniProt DB are dropped silently.

    Abundance for each surviving row is computed as
    ``level * uniprot_db.mw[matched_index]``. The returned
    ``abundances`` array is 1-D since paxDB files contain a single
    column of levels; callers can stack multiple ``ProtData`` objects
    column-wise to form a 2-D array if they want multi-sample
    averaging.

    Ported from GECKO MATLAB:
    the inline paxDB-parsing block of
    src/limit_proteins/calculateFfactor.m.

    Parameters
    ----------
    path
        Path to ``paxDB.tsv``.
    uniprot_db
        Loaded UniProt database, used to resolve gene names to
        UniProt IDs and look up molecular weights.

    Returns
    -------
    ProtData

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"paxDB file not found: {path}")

    text = path.read_text(encoding="utf-8")

    gene_to_index: dict[str, int] = {}
    for i, name in enumerate(uniprot_db.genes):
        if name:
            gene_to_index.setdefault(name, i)

    uniprot_ids: list[str] = []
    abundances: list[float] = []

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        gene_raw = parts[1].strip()
        try:
            level = float(parts[2])
        except ValueError:
            continue
        gene_id = _GENE_PREFIX_RE.sub("", gene_raw)
        idx = gene_to_index.get(gene_id)
        if idx is None:
            continue
        uniprot_ids.append(uniprot_db.ids[idx])
        mw = float(uniprot_db.mw[idx])
        abundances.append(level * mw)

    return ProtData(
        uniprot_ids=uniprot_ids,
        abundances=np.array(abundances, dtype=float),
    )
