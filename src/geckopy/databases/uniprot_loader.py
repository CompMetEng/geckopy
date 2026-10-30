"""Parse uniprot.tsv into a UniprotDB struct.

Ported from GECKO MATLAB: src/geckomat/get_enzyme_data/loadDatabases.m
(the UniProt-parsing branch; the download branch is deferred to a
separate future module).

Schema expected in uniprot.tsv (tab-delimited, one header line):

    Entry | Gene Names | EC number | Mass | Sequence

All five columns are always present, though individual cells may be
empty. Mass is in Da in the file and stored as Da in the UniprotDB to match
GECKO's ec.mw convention. Duplicate Entry values raise ValueError,
matching MATLAB's dispEM-based hard failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class UniprotDB:
    """Parsed uniprot.tsv. Arrays are aligned: row i in every field
    describes the same database entry (or gene-name row when split).

    ``mw`` is in Da, matching the Mass column of the TSV verbatim.
    """
    ids: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    eccodes: list[str] = field(default_factory=list)
    mw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    sequences: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def find_by_gene(self, gene: str) -> int | None:
        """Return the row index of the first entry whose gene matches, or None."""
        try:
            return self.genes.index(gene)
        except ValueError:
            return None

    def find_by_id(self, uniprot_id: str) -> int | None:
        """Return the row index of the first entry with this UniProt ID, or None."""
        try:
            return self.ids.index(uniprot_id)
        except ValueError:
            return None


def load_uniprot_tsv(
    path: str | Path,
    *,
    split_gene_cells: bool = False,
    uniprot_id: str | int | None = None,
    id_type: str = "taxonomy_id",
    gene_id_field: str = "gene_oln",
    reviewed: bool = True,
    auto_download: bool = True,
) -> UniprotDB:
    """Parse a uniprot.tsv file into a UniprotDB.

    If ``path`` does not exist and ``uniprot_id`` is given and
    ``auto_download`` is True, ``download_uniprot`` is invoked first.

    Parameters
    ----------
    path
        Path to uniprot.tsv. May not exist if ``uniprot_id`` triggers
        an auto-download.
    split_gene_cells
        If True, rows whose Gene Names cell contains whitespace-separated
        multiple names are expanded into one row per gene name. All other
        fields (Entry, EC, MW, sequence) are duplicated across the
        expanded rows. This matters for organisms where the UniProt
        ``gene_oln`` field returns multiple space-separated ORF names
        per entry: with the default of False, the cell is stored
        verbatim (including any embedded whitespace), so an exact-match
        lookup against a single ORF name silently misses those genes.
    uniprot_id
        UniProt-side identifier (e.g. NCBI taxonomy ID). Required
        for auto-download; otherwise unused.
    id_type
        UniProt query field type for ``uniprot_id``. Default
        ``"taxonomy_id"``.
    gene_id_field
        UniProt field that becomes the Gene Names column. Default
        ``"gene_oln"``.
    reviewed
        If True (default), restrict the auto-download query to
        reviewed (Swiss-Prot) entries.
    auto_download
        If True (default) and ``path`` is missing, trigger
        ``download_uniprot`` before parsing. Set to False to keep
        the loader pure (e.g. tests).

    Returns
    -------
    UniprotDB
        Parsed database. MW is in Da.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist and either ``auto_download`` is
        False or ``uniprot_id`` is None.
    ValueError
        If the file has duplicate Entry values, or if the header does
        not have at least 5 tab-delimited fields.

    Notes
    -----
    Empty cells (common for EC number and Sequence) become empty
    strings. Cells are read as :func:`read_quoted_tsv` reads them, as
    MATLAB's ``textscan`` with ``%q`` does: the quotes around a cell
    are dropped, so ``"1.1.1.25; 2.5.1.19"`` gives ``1.1.1.25; 2.5.1.19``.
    Duplicate entries raise, matching MATLAB's dispEM failure mode.
    """
    path = Path(path)
    if not path.is_file():
        if not auto_download or uniprot_id is None:
            raise FileNotFoundError(
                f"uniprot.tsv not found: {path}. "
                f"Pass uniprot_id=... to trigger an auto-download, "
                f"or run `geckopy uniprot-download` first."
            )
        from .uniprot_download import download_uniprot
        download_uniprot(
            uniprot_id, path,
            id_type=id_type,
            gene_id_field=gene_id_field,
            reviewed=reviewed,
        )

    ids: list[str] = []
    genes: list[str] = []
    eccodes: list[str] = []
    mw_da: list[float] = []
    sequences: list[str] = []
    # One Entry per source row (before gene-cell splitting), so duplicate
    # detection sees each row exactly once regardless of how many genes it
    # expands to.
    source_row_ids: list[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if not text:
        raise ValueError(f"{path} is empty.")
    header_end = _first_line_end(text)
    header_fields = text[:header_end].rstrip("\r\n").split("\t")
    if len(header_fields) < 5:
        raise ValueError(
            f"{path} header has {len(header_fields)} fields, expected 5."
        )

    for entry, gene_cell, ec, mass_str, seq in read_quoted_tsv(text, 5, header_lines=1):
        # MATLAB str2double returns NaN for empty or non-numeric; mirror.
        try:
            mass_da = float(mass_str) if mass_str.strip() else float("nan")
        except ValueError:
            mass_da = float("nan")

        if split_gene_cells and gene_cell.strip():
            gene_names = gene_cell.split()
        else:
            gene_names = [gene_cell]

        source_row_ids.append(entry)
        for gene_name in gene_names:
            ids.append(entry)
            genes.append(gene_name)
            eccodes.append(ec)
            mw_da.append(mass_da)
            sequences.append(seq)

    _check_duplicate_ids(source_row_ids, path)

    return UniprotDB(
        ids=ids,
        genes=genes,
        eccodes=eccodes,
        mw=np.array(mw_da, dtype=float),
        sequences=sequences,
    )


def _check_duplicate_ids(source_row_ids: list[str], path: Path) -> None:
    """Raise if any Entry value appears on more than one source row.

    ``source_row_ids`` holds one Entry per source row (gene-cell
    splitting already accounted for), so a plain count flags genuine
    cross-row duplicates without mistaking a split row for one.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in source_row_ids:
        if entry in seen:
            duplicates.append(entry)
        seen.add(entry)

    if duplicates:
        preview = sorted(set(duplicates))[:5]
        raise ValueError(
            f"Duplicate Entry values in {path}: {preview}. "
            f"Manually curate uniprot.tsv or adjust adapter parameters."
        )


def _first_line_end(text: str, start: int = 0) -> int:
    """Index just after the end of the line of ``text`` that begins at
    ``start`` (the length of ``text`` for the last line)."""
    for i in range(start, len(text)):
        c = text[i]
        if c == "\n":
            return i + 1
        if c == "\r":
            return i + 2 if text[i + 1:i + 2] == "\n" else i + 1
    return len(text)


def read_quoted_tsv(text: str, n_columns: int, header_lines: int = 0) -> list[list[str]]:
    """Split tab-delimited ``text`` into rows of ``n_columns`` cells the way
    MATLAB's ``textscan(fid, '%q ...', 'Delimiter', '\\t')`` does.

    A cell that starts with a double quote (after spaces) ends at the next
    double quote that is not doubled; a doubled quote inside stands for one
    quote, and tabs and line breaks inside are part of the cell. Text between
    the closing quote and the next tab or line break is appended to the cell.
    A quote that is never closed takes the rest of the text. In any other
    cell the quotes are kept, and leading spaces are dropped. A line break
    ends the row, so a short line is padded with empty cells; the cells of a
    line beyond ``n_columns`` start a new row. A line without any text is
    skipped. The first ``header_lines`` lines are skipped as they are, as
    ``textscan`` does with ``'HeaderLines'``.
    """
    rows: list[list[str]] = []
    row: list[str] = []
    n = len(text)
    i = 0
    for _ in range(header_lines):
        i = _first_line_end(text, i)

    def finish(cell: str, at_line_end: bool) -> None:
        row.append(cell)
        if len(row) == n_columns or at_line_end:
            row.extend([""] * (n_columns - len(row)))
            rows.append(row.copy())
            row.clear()

    while i < n:
        while i < n and text[i] in " \b":
            i += 1
        quoted = i < n and text[i] == '"'
        cell: list[str] = []
        if quoted:
            i += 1
            while i < n:
                if text[i] == '"':
                    if text[i + 1:i + 2] == '"':
                        cell.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                cell.append(text[i])
                i += 1
        while i < n and text[i] not in "\t\r\n":
            cell.append(text[i])
            i += 1
        value = "".join(cell)
        if i >= n:
            if value or quoted or row:
                finish(value, True)
            break
        c = text[i]
        if c == "\t":
            i += 1
            finish(value, False)
            if i >= n and row:
                finish("", True)
            continue
        i += 2 if c == "\r" and text[i + 1:i + 2] == "\n" else 1
        if value or quoted or row:
            finish(value, True)
    return rows
