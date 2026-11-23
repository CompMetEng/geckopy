"""Load the DLKcat ignore-list TSV files.

Ported from GECKO MATLAB:
the inline `DLKcatIgnoreMets.tsv` and `DLKcatCurrencyMets.tsv` parsing
in src/gather_kcats/writeDLKcatInput.m.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


# Match MATLAB `lower(regexprep(name, '[^0-9a-zA-Z]+', ''))`.
_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z]+")

# Column-label tokens used to detect (and skip) an optional header row. Real
# data cells (e.g. "H2O", "ATP", a SMILES string) never match these, so a
# header-less file is unaffected.
_HEADER_TOKENS = frozenset({
    "name", "names", "met", "mets", "metname", "metnames", "metabolite",
    "metabolites", "metabolite1", "metabolite2", "met1", "met2", "compound",
    "compounds", "smiles", "substrate", "product",
})


def _normalize(name: str) -> str:
    """Lowercase + strip non-alphanumeric characters."""
    return _NORMALIZE_RE.sub("", name).lower()


@dataclass
class DLKcatIgnoreLists:
    """Lists of metabolites to skip when writing DLKcat input.

    Attributes
    ----------
    ignore_names
        Normalized (lowercase, alphanumeric-only) metabolite names to
        skip entirely.
    ignore_smiles
        SMILES strings to skip; matched as-is against
        ``metabolite.annotation['smiles']``.
    currency_pairs
        Pairs of normalized metabolite names. When both halves of a
        pair appear in the same reaction, both are stripped from
        that reaction's substrates (unless removal would leave no
        substrates).
    """

    ignore_names: list[str] = field(default_factory=list)
    ignore_smiles: list[str] = field(default_factory=list)
    currency_pairs: list[tuple[str, str]] = field(default_factory=list)


def load_dlkcat_ignore_lists(
    project_data_folder: Path | str | None = None,
) -> DLKcatIgnoreLists:
    """Load the two DLKcat ignore-list TSV files.

    Looks for ``DLKcatIgnoreMets.tsv`` and ``DLKcatCurrencyMets.tsv``
    in ``project_data_folder`` first; falls back to the defaults
    included in ``geckopy.data`` for any file missing from the
    project folder.

    The two files are independent: the project folder may override
    just one of them.

    Ported from GECKO MATLAB:
    the inline file-loading parts of
    src/gather_kcats/writeDLKcatInput.m.

    Parameters
    ----------
    project_data_folder
        Project's ``data/`` folder, typically
        ``adapter.params.path / "data"``. If ``None`` (or the file
        is absent there), the default included with geckopy is used.

    Returns
    -------
    DLKcatIgnoreLists
        Normalized ignore lists ready to feed to
        ``kcats_from_dlkcat``.
    """
    ignore_lines = _strip_header(_read_lines(
        _resolve_path(project_data_folder, "DLKcatIgnoreMets.tsv")
    ))
    currency_lines = _strip_header(_read_lines(
        _resolve_path(project_data_folder, "DLKcatCurrencyMets.tsv")
    ))

    ignore_names: list[str] = []
    ignore_smiles: list[str] = []
    for line in ignore_lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, smiles = parts[0].strip(), parts[1].strip()
        if name:
            ignore_names.append(_normalize(name))
        if smiles:
            ignore_smiles.append(smiles)

    currency_pairs: list[tuple[str, str]] = []
    for line in currency_lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a and b:
            currency_pairs.append((_normalize(a), _normalize(b)))

    return DLKcatIgnoreLists(
        ignore_names=ignore_names,
        ignore_smiles=ignore_smiles,
        currency_pairs=currency_pairs,
    )


def _resolve_path(project_data_folder, filename: str):
    """Project file if present; else the included default."""
    if project_data_folder is not None:
        candidate = Path(project_data_folder) / filename
        if candidate.is_file():
            return candidate
    return resources.files("geckopy.data").joinpath(filename)


def _read_lines(path) -> list[str]:
    """Read non-empty stripped lines from a Path or importlib Traversable."""
    text = path.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def _strip_header(lines: list[str]) -> list[str]:
    """Drop a leading header row if the first line's cells are column labels."""
    if lines:
        first_cells = [c.strip().lower() for c in lines[0].split("\t")[:2]]
        if any(c in _HEADER_TOKENS for c in first_cells if c):
            return lines[1:]
    return lines
