"""Parse ComplexPortal.json into a list of ComplexPortalEntry.

Ported from GECKO MATLAB:
src/geckomat/change_model/applyComplexData.m (the JSON-loading branch).
The download branch is deferred to a future module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ComplexPortalEntry:
    """One ComplexPortal complex.

    The JSON field name ``stochiometry`` (a typo in the original
    ComplexPortal API) is renamed to ``stoichiometry`` here. The
    field ``geneName`` is renamed to ``gene_names`` for Python
    conventions. Both renames happen only on this in-memory object;
    the JSON file itself is untouched.
    """
    complex_id: str
    name: str
    species: str
    gene_names: list[str] = field(default_factory=list)
    protein_ids: list[str] = field(default_factory=list)
    stoichiometry: list[int] = field(default_factory=list)


def load_complex_portal_json(path: str | Path) -> list[ComplexPortalEntry]:
    """Parse a ComplexPortal JSON file into a list of entries.

    Parameters
    ----------
    path
        Path to a ComplexPortal.json file. The file is expected to be
        a JSON array of complex objects with these fields:
        ``complexID``, ``name``, ``specie``, ``geneName``, ``protID``,
        ``stochiometry``.

    Returns
    -------
    list of ComplexPortalEntry

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or any entry is malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ComplexPortal.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from None

    # ComplexPortal exports are arrays of objects; tolerate a single
    # object too, matching MATLAB's jsondecode flexibility.
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected a JSON array of complexes, got {type(raw).__name__}"
        )

    entries: list[ComplexPortalEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path}: entry {i} is not a JSON object"
            )
        try:
            entries.append(ComplexPortalEntry(
                complex_id=item["complexID"],
                name=item.get("name", ""),
                species=item.get("specie", ""),
                gene_names=_as_list(item.get("geneName", [])),
                protein_ids=_as_list(item.get("protID", [])),
                stoichiometry=[
                    int(s) for s in _as_list(item.get("stochiometry", []))
                ],
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{path}: entry {i} is malformed ({e})"
            ) from None

    return entries


def _as_list(value):
    """Wrap a scalar in a single-element list; pass lists through.

    ComplexPortal exports collapse single-element arrays to scalars
    (matching the behaviour of MATLAB's `jsonencode` when the file is written and read back); this
    normalises them back to lists.
    """
    if isinstance(value, list):
        return value
    return [value]
