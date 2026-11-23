"""Walk the BRENDA bulk JSON and yield one Row per measurement.

JSON shape per ``https://www.brenda-enzymes.org/schemas/2.0.0/enzyme.schema.json``:
``data[<ec>].{turnover_number, specific_activity, molecular_weight}``
each holds a list of ``numeric_dataset`` entries (``value`` string,
``comment``, ``proteins[]``, ``references[]``). Substrate is embedded
in the ``value`` string as ``"23.5 {ethanol}"``. Organism lives in
``data[<ec>].protein[<pid>].organism``. PMIDs live in
``data[<ec>].reference[<rid>].pmid``.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Literal, NamedTuple

logger = logging.getLogger(__name__)


# -999 is BRENDA's missing-value marker.
_MISSING_VALUE = -999.0

# Bar-Even et al. 2011, Biochemistry 50:4402 - physical upper bound for kcat.
_MAX_PHYSICAL_KCAT = 1e7

# Upper bound for a protein molecular weight in Da (~10 MDa, well above any
# single chain). A parse error yielding e.g. 1e9 would otherwise flow into
# mw.tsv and corrupt the SA-derived kcats that multiply by it.
_MAX_PHYSICAL_MW = 1e7

# Matches "23.5", "0.1-2.5", "1e-3", optionally followed by " {substrate}".
_VALUE_RE = re.compile(
    r"""
    ^\s*
    (?P<low>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
    (?:\s*-\s*
       (?P<high>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
    )?
    \s*
    (?:\{(?P<substrate>[^}]*)\})?
    \s*$
    """,
    re.VERBOSE,
)

_MUTANT_RE = re.compile(r"mutant|mutated", re.IGNORECASE)

Kind = Literal["kcat", "sa", "mw"]

_FIELD_BY_KIND: dict[Kind, str] = {
    "kcat": "turnover_number",
    "sa": "specific_activity",
    "mw": "molecular_weight",
}


class Row(NamedTuple):
    """One measurement yielded by ``parse_brenda_json``."""

    kind: Kind
    ec: str
    substrate: str
    organism: str
    value: float
    references: tuple[str, ...]


def parse_brenda_json(path: str | Path) -> Iterator[Row]:
    """Yield one ``Row`` per (kind, EC, substrate, organism) measurement.

    Range values like ``"0.1-2.5 {NADH}"`` collapse to their upper
    bound. Mutant-flagged rows, the ``-999`` missing-value marker, kcat values
    above the physical ceiling, and the ``spontaneous`` pseudo-EC are
    skipped. Measurements fan out across ``proteins[]`` so one input
    row may produce several output rows (one per organism).

    Parameters
    ----------
    path
        Path to the unpacked BRENDA bulk JSON file
        (Release 2026.1 or later, schema version 2.0.0).

    Yields
    ------
    Row
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    yield from _walk_doc(doc)


def _walk_doc(doc: dict) -> Iterator[Row]:
    data = doc.get("data", {})
    for ec, entry in data.items():
        if ec == "spontaneous":
            continue
        if not isinstance(entry, dict):
            continue
        proteins_map = entry.get("protein", {}) or {}
        pmid_lookup = _build_pmid_lookup(entry.get("reference", {}) or {})
        for kind, field_name in _FIELD_BY_KIND.items():
            for raw in entry.get(field_name, []) or []:
                yield from _parse_one(kind, ec, raw, proteins_map, pmid_lookup)


def _build_pmid_lookup(reference_map: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for rid, ref in reference_map.items():
        if not isinstance(ref, dict):
            continue
        pmid = ref.get("pmid")
        if isinstance(pmid, int):
            out[rid] = f"PMID:{pmid}"
    return out


def _parse_one(
    kind: Kind,
    ec: str,
    raw: dict,
    proteins_map: dict,
    pmid_lookup: dict[str, str],
) -> Iterator[Row]:
    if not isinstance(raw, dict):
        return
    if _MUTANT_RE.search(raw.get("comment", "") or ""):
        return
    value_str = raw.get("value", "") or ""
    match = _VALUE_RE.match(value_str)
    if match is None:
        logger.debug("%s/%s: unparseable value %r", ec, kind, value_str)
        return
    high = match["high"]
    value = float(high) if high is not None else float(match["low"])
    if value == _MISSING_VALUE:
        return
    if kind == "kcat" and value > _MAX_PHYSICAL_KCAT:
        return
    # SA and MW must be positive; an absurd MW would corrupt SA-derived kcats.
    if kind in ("sa", "mw") and value <= 0:
        return
    if kind == "mw" and value > _MAX_PHYSICAL_MW:
        return

    substrate_raw = match["substrate"]
    if kind == "kcat" and substrate_raw:
        substrate = _normalize(substrate_raw)
    else:
        substrate = "*"

    refs = tuple(sorted({
        pmid_lookup[rid]
        for rid in raw.get("references", []) or []
        if rid in pmid_lookup
    }))

    for pid in raw.get("proteins", []) or []:
        # The "protein" map is keyed by string (JSON object keys always are),
        # while the "proteins" reference list may contain ints; coerce so the
        # lookup matches regardless and the measurement is not silently lost.
        protein = proteins_map.get(str(pid))
        if not isinstance(protein, dict):
            continue
        org = (protein.get("organism") or "").strip()
        if not org:
            continue
        yield Row(kind, ec, substrate, _normalize(org), value, refs)


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())
