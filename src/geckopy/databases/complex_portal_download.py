"""Download complex stoichiometries from EBI Complex Portal.

Ported from GECKO MATLAB:
src/geckomat/change_model/getComplexData.m.

Network access is required. The result is cached as
``ComplexPortal.json`` in the adapter's data folder; subsequent runs
should call ``load_complex_portal_json`` instead of this function
unless explicitly refreshing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .complex_portal_loader import ComplexPortalEntry


logger = logging.getLogger(__name__)

# TODO: progress reporting (tqdm) deferred per design discussion;
# revisit when downloaders are integrated end-to-end.

_SEARCH_URL = "https://www.ebi.ac.uk/intact/complex-ws/search/*"
_DETAIL_URL = "https://www.ebi.ac.uk/intact/complex-ws/complex/"
_TIMEOUT_SECONDS = 30
_NUM_RETRIES = 10
_SEARCH_PAGE_SIZE = 100
# Safety bound: 100 pages * 100 = 10k complexes, well above any organism.
_MAX_SEARCH_PAGES = 100


def get_complex_data(
    taxonomic_id: Optional[int],
    *,
    write_to: Optional[Path] = None,
) -> list[ComplexPortalEntry]:
    """Download complex stoichiometries from EBI Complex Portal.

    Ported from GECKO MATLAB:
    src/geckomat/change_model/getComplexData.m.

    Taxonomic ID semantics:

    - A positive integer queries that organism only.
    - ``0`` queries all organisms (no species filter).
    - ``None`` raises ValueError.

    The response is post-processed to expand "complexes of complexes":
    when a Complex Portal entry references other complexes as subunits,
    those are flattened into one combined protein list with multiplied
    stoichiometries.

    Parameters
    ----------
    taxonomic_id
        NCBI taxonomy ID for the organism, or 0 for all organisms.
    write_to
        Optional path. If given, the data is also written as JSON
        (matching the MATLAB ``ComplexPortal.json`` format) to this
        path so subsequent runs can use ``load_complex_portal_json``.

    Returns
    -------
    list of ComplexPortalEntry
        Parsed and post-processed complexes.

    Raises
    ------
    ValueError
        If ``taxonomic_id`` is None.
    requests.HTTPError
        If the API returns an unrecoverable error.
    """
    if taxonomic_id is None:
        raise ValueError(
            "taxonomic_id is required. Use 0 to query all organisms, "
            "or set adapter.params.complex.taxonomic_id."
        )

    session = _make_session()

    # Step 1: search to enumerate complex IDs. The search endpoint paginates,
    # so a single request silently truncates organisms with more than one
    # page of complexes. ``first`` is a record offset and ``number`` the page
    # length; the grand total is ``totalNumberOfResults`` (``size`` counts
    # only the elements in the page at hand).
    search_url = _SEARCH_URL
    if taxonomic_id != 0:
        search_url = (
            f"{_SEARCH_URL}?facets=species&filters=species:(\"{taxonomic_id}\")"
        )
    logger.info(
        "Querying Complex Portal for taxonomic_id=%s ...", taxonomic_id
    )

    complex_ids: list[str] = []
    total: Optional[int] = None
    for page in range(_MAX_SEARCH_PAGES):
        response = session.get(
            search_url,
            params={
                "first": page * _SEARCH_PAGE_SIZE,
                "number": _SEARCH_PAGE_SIZE,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        search_data = response.json()

        if total is None:
            total = int(search_data.get("totalNumberOfResults", 0))
            if total == 0:
                raise ValueError(
                    f"No complexes returned for taxonomic_id={taxonomic_id}."
                )

        elements = search_data.get("elements") or []
        if not elements:
            break
        complex_ids.extend(elem["complexAC"] for elem in elements)
        if len(complex_ids) >= total:
            break
    if total is not None and len(complex_ids) < total:
        logger.warning(
            "Complex Portal reported %d complexes but only %d were "
            "collected before the %d-page bound.",
            total, len(complex_ids), _MAX_SEARCH_PAGES,
        )
    logger.info(
        "Found %d complexes; fetching details ...", len(complex_ids)
    )

    # Step 2: fetch details for each complex.
    raw_complexes: list[dict] = []
    for cid in complex_ids:
        try:
            detail = _fetch_complex_detail(session, cid)
        except requests.HTTPError as e:
            logger.warning("Cannot retrieve %s: %s", cid, e)
            continue
        if detail is not None:
            raw_complexes.append(detail)

    # Step 3: post-process (expand sub-complexes, build the structured form).
    structured = _post_process(raw_complexes)

    # Step 4: optional write to disk.
    if write_to is not None:
        write_to = Path(write_to)
        write_to.parent.mkdir(parents=True, exist_ok=True)
        _write_json(structured, write_to)
        logger.info("ComplexPortal cache written to %s", write_to)

    # Convert to ComplexPortalEntry list.
    return [
        ComplexPortalEntry(
            complex_id=row["complexID"],
            name=row.get("name", ""),
            species=row.get("species", ""),
            gene_names=list(row.get("geneName", [])),
            protein_ids=list(row.get("protID", [])),
            stoichiometry=[int(s) for s in row.get("stochiometry", [])],
        )
        for row in structured
    ]


def _make_session() -> requests.Session:
    """Build a requests.Session with retry/backoff configured."""
    session = requests.Session()
    retry = Retry(
        total=_NUM_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_complex_detail(
    session: requests.Session, complex_id: str,
) -> Optional[dict]:
    """Fetch one complex's detailed record. Returns None on 404."""
    url = _DETAIL_URL + complex_id
    try:
        response = session.get(url, timeout=_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except requests.HTTPError:
        return None

    payload = response.json()
    return _parse_complex_detail(payload)


def _parse_complex_detail(payload: dict) -> dict:
    """Convert one detail JSON into the row-shaped dict used downstream."""
    # Minimum validation: required fields per Q5 design.
    if "complexAc" not in payload:
        raise ValueError("Complex Portal response missing 'complexAc' field.")

    participants = payload.get("participants", []) or []
    protein_participants = [
        p for p in participants
        if str(p.get("interactorType", "")).lower() == "protein"
    ]
    # A sub-complex participant has an interactorType naming a complex
    # ("complex", "stable complex", ...). Classifying by this rather than
    # by "no proteins present" stops complexes whose only non-protein
    # participants are ligands/RNA from being mistaken for complexes-of-
    # complexes and silently reduced to zero proteins in _post_process.
    complex_participants = [
        p for p in participants
        if "complex" in str(p.get("interactorType", "")).lower()
    ]

    if protein_participants:
        eligible = protein_participants
        defined = 1
    elif complex_participants:
        eligible = complex_participants
        defined = 2
    else:
        # Only ligands / nucleic acids / etc. -- nothing usable as an enzyme.
        eligible = []
        defined = 0

    gene_names: list[str] = []
    protein_ids: list[str] = []
    stoichiometries: list[int] = []
    any_stoich = False

    for p in eligible:
        gene_names.append(p.get("name", ""))
        protein_ids.append(p.get("identifier", ""))
        s = p.get("stochiometry")
        if s in (None, ""):
            stoichiometries.append(0)
        else:
            any_stoich = True
            # Format is "minValue: <n>, maxValue: <m>"; we want minValue.
            stoichiometries.append(_parse_stoich_minvalue(s))

    if not any_stoich:
        stoichiometries = [0] * len(eligible)
        defined = 0

    return {
        "complexID": payload["complexAc"],
        "name": payload.get("name", ""),
        "species": payload.get("species", ""),
        "geneName": gene_names,
        "protID": protein_ids,
        "stochiometry": stoichiometries,
        "defined": defined,
    }


def _parse_stoich_minvalue(s: str) -> int:
    """Parse 'minValue: 2, maxValue: 4' to 2."""
    s = str(s)
    parts = s.split(",")
    first = parts[0].strip()
    if ":" in first:
        first = first.split(":", 1)[1].strip()
    try:
        return int(first)
    except ValueError:
        return 0


def _post_process(rows: list[dict]) -> list[dict]:
    """Expand 'complexes of complexes' into flat protein lists.

    A row with ``defined == 2`` has sub-complexes as subunits; replace
    its proteins with the union of subunits' proteins, multiplied by
    the row's stoichiometries.
    """
    by_id = {row["complexID"]: row for row in rows}

    for row in rows:
        if row.get("defined") != 2:
            continue
        sub_ids = row["protID"]
        sub_stoichs = row["stochiometry"]

        merged_genes: list[str] = []
        merged_prots: list[str] = []
        merged_stoichs: list[int] = []

        for sub_id, multiplier in zip(sub_ids, sub_stoichs):
            sub = by_id.get(sub_id)
            if sub is None:
                continue
            for g, p, s in zip(
                sub["geneName"], sub["protID"], sub["stochiometry"],
            ):
                merged_genes.append(g)
                merged_prots.append(p)
                merged_stoichs.append(s * multiplier)

        # Deduplicate by protein ID, summing stoichiometries.
        seen: dict[str, tuple[int, str]] = {}
        for g, p, s in zip(merged_genes, merged_prots, merged_stoichs):
            if p in seen:
                idx, _ = seen[p]
                merged_stoichs[idx] += s
            else:
                seen[p] = (len(seen), g)

        # Rebuild deduplicated lists in insertion order.
        unique_prots = list(seen.keys())
        unique_genes = [seen[p][1] for p in unique_prots]
        unique_stoichs: list[int] = []
        accum: dict[str, int] = {p: 0 for p in unique_prots}
        for p, s in zip(merged_prots, merged_stoichs):
            accum[p] += s
        unique_stoichs = [accum[p] for p in unique_prots]

        row["geneName"] = unique_genes
        row["protID"] = unique_prots
        row["stochiometry"] = unique_stoichs

    return rows


def _write_json(rows: list[dict], path: Path) -> None:
    """Write rows in the same shape used by load_complex_portal_json."""
    payload = [
        {
            "complexID": r["complexID"],
            "name": r.get("name", ""),
            "specie": r.get("species", ""),
            "geneName": list(r.get("geneName", [])),
            "protID": list(r.get("protID", [])),
            "stochiometry": list(r.get("stochiometry", [])),
        }
        for r in rows
    ]
    path.write_text(json.dumps(payload, indent=2))
