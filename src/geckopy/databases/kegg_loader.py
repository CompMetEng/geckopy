"""Parse kegg.tsv into a KeggDB struct, downloading from KEGG REST if missing.

Ported from GECKO MATLAB:
src/download_databases/loadDatabases.m (KEGG branch).

Schema in kegg.tsv (comma-delimited, no header, 7 columns):

    uniprot_id, gene_name, kegg_gene, eccodes, mw, pathway, sequence

The ``gene_name`` column (col 2) is the matching key. By default it
equals ``kegg_gene`` (col 3); the adapter's ``kegg.gene_id_field``
can route a different KEGG field there.

``ec.enzymes`` falls back to ``kegg_gene`` (the bare KEGG gene ID,
without the ``orgcode:`` prefix) when ``uniprot_id`` is empty.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeggDB:
    """Parsed kegg.tsv. Arrays are aligned: row i in every field
    describes the same KEGG entry.

    ``mw`` is in Da, matching the file column verbatim.
    """
    uniprot_ids: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    kegg_genes: list[str] = field(default_factory=list)
    eccodes: list[str] = field(default_factory=list)
    mw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    pathways: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.genes)

    def find_by_gene(self, gene: str) -> int | None:
        """Return the row index of the first entry whose gene matches, or None."""
        try:
            return self.genes.index(gene)
        except ValueError:
            return None


def load_kegg_tsv(
    path: str | Path,
    *,
    kegg_id: str | None = None,
    gene_id_field: str = "kegg",
    auto_download: bool = True,
) -> KeggDB | None:
    """Parse a kegg.tsv into a KeggDB.

    If ``path`` does not exist and ``kegg_id`` is given and
    ``auto_download`` is True, fetches the data via ``download_kegg``
    first, then parses. An empty ``kegg_id`` skips the download with a
    warning and returns None, as GECKO's ``loadDatabases`` does without a
    ``kegg.ID``.

    Parameters
    ----------
    path
        Path to kegg.tsv. May not exist if ``kegg_id`` triggers an
        auto-download.
    kegg_id
        KEGG organism code (e.g. ``"sce"``). Required for
        auto-download; otherwise unused. ``""`` means no KEGG data.
    gene_id_field
        KEGG entry field to use as the gene matching key. Default
        ``"kegg"`` uses column 2, the bare KEGG gene ID.
    auto_download
        If True (default) and ``path`` is missing, trigger
        ``download_kegg(kegg_id, path, gene_id_field=...)`` before
        parsing. Set to False to keep the loader pure (e.g. tests).

    Returns
    -------
    KeggDB or None
        None when ``path`` does not exist and ``kegg_id`` is empty.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist and either ``auto_download`` is
        False or ``kegg_id`` is None.
    ValueError
        If the file has malformed rows.
    """
    path = Path(path)
    if not path.is_file():
        if auto_download and kegg_id == "":
            logger.warning("No KEGG organism code (kegg_id) is given; KEGG data is not downloaded.")
            return None
        if not auto_download or kegg_id is None:
            raise FileNotFoundError(
                f"kegg.tsv not found: {path}. "
                f"Pass kegg_id=... to trigger an auto-download, "
                f"or run `geckopy kegg-download` first."
            )
        from .kegg_download import download_kegg
        download_kegg(kegg_id, path, gene_id_field=gene_id_field)

    uniprot_ids: list[str] = []
    genes: list[str] = []
    kegg_genes: list[str] = []
    eccodes: list[str] = []
    mw_da: list[float] = []
    pathways: list[str] = []
    sequences: list[str] = []

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            while len(row) < 7:
                row.append("")
            uni, gene, kgene, ec, mw_str, pathway, seq = row[:7]
            try:
                mw = float(mw_str) if mw_str.strip() else float("nan")
            except ValueError:
                mw = float("nan")
                logger.warning(
                    "%s:%d: MW column %r is not numeric; using NaN.",
                    path.name, line_no, mw_str,
                )
            uniprot_ids.append(uni)
            genes.append(gene)
            kegg_genes.append(kgene)
            eccodes.append(ec)
            mw_da.append(mw)
            pathways.append(pathway)
            sequences.append(seq)

    return KeggDB(
        uniprot_ids=uniprot_ids,
        genes=genes,
        kegg_genes=kegg_genes,
        eccodes=eccodes,
        mw=np.array(mw_da, dtype=float),
        pathways=pathways,
        sequences=sequences,
    )
