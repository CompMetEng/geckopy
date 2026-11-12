"""Download organism-specific protein information from KEGG REST.

Ported from GECKO MATLAB:
src/download_databases/loadDatabases.m
(the KEGG-download branch, extracted into its own module).

Two REST endpoints are used:

* ``https://rest.kegg.jp/list/<org>`` returns the gene list, one
  ``<org>:<gene>\\t<description>`` per line.
* ``https://rest.kegg.jp/get/<org>:g1+<org>:g2+...`` returns full
  KEGG entries in the flat-file format (10 genes per request).

For each entry we extract: UniProt accession, gene name (the field
chosen by ``gene_id_field``), bare KEGG gene ID, EC numbers,
pathway list, and the AA sequence. MW is computed from the
sequence via ``calculate_mw``. Entries missing either AASEQ or
UniProt are dropped. Output is a comma-delimited TSV with no
header, matching the schema consumed by ``load_kegg_tsv``.
"""
from __future__ import annotations

import csv
import logging
import re
import time
from pathlib import Path

import requests

from .mw import calculate_mw

logger = logging.getLogger(__name__)


_LIST_URL = "https://rest.kegg.jp/list/{org}"
_GET_URL = "https://rest.kegg.jp/get/{query}"

_GENES_PER_QUERY = 10
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 5
_RETRY_BACKOFF = 1.0
# KEGG asks for at most 3 requests/second; ~0.34 s between calls stays under.
_INTER_REQUEST_SLEEP = 0.34


def download_kegg(
    kegg_id: str,
    out_path: str | Path,
    *,
    gene_id_field: str = "kegg",
    session: requests.Session | None = None,
) -> Path:
    """Download per-gene KEGG data for ``kegg_id`` and write to ``out_path``.

    Parameters
    ----------
    kegg_id
        KEGG organism code, e.g. ``"sce"``.
    out_path
        Destination CSV path (overwritten if it exists).
    gene_id_field
        KEGG entry field to use as the gene matching key in col 2.
        ``"kegg"`` (default) copies the bare KEGG gene ID from
        ``ENTRY``. Other values name an alternative annotation line
        (e.g. ``"OrderedLocus"``), which is matched on the prefix.
    session
        Optional requests.Session for connection reuse and test
        injection. A new transient session is created if absent.

    Returns
    -------
    Path
        The written ``out_path``.

    Raises
    ------
    RuntimeError
        If the organism list cannot be retrieved or a KEGG request is
        rejected (e.g. an invalid organism code). Batches that come back
        with fewer entries than requested are not an error; the missing
        gene IDs are skipped with a debug log.
    ValueError
        If ``gene_id_field`` is not present in any returned entry.
    """
    out_path = Path(out_path)
    sess = session or requests.Session()

    gene_ids = _list_genes(sess, kegg_id)
    if not gene_ids:
        raise RuntimeError(
            f"KEGG returned no genes for organism code {kegg_id!r}; "
            f"verify the code at https://rest.kegg.jp/list/organism."
        )
    logger.info("KEGG %s: %d genes to fetch", kegg_id, len(gene_ids))

    entries: list[dict[str, str]] = []
    for batch_no, batch in enumerate(_chunks(gene_ids, _GENES_PER_QUERY)):
        if batch_no > 0:
            # KEGG asks for <=3 requests/s; throttle between batches.
            time.sleep(_INTER_REQUEST_SLEEP)
        text = _fetch_get(sess, kegg_id, batch)
        raw_entries = _split_entries(text)
        if len(raw_entries) < len(batch):
            # KEGG legitimately omits entries for withdrawn/obsolete gene
            # IDs, so a short batch is expected, not an error. Entries are
            # matched by their own ENTRY line in _parse_entry, not by
            # position, so the missing ones simply drop out.
            logger.debug(
                "KEGG returned %d/%d entries for a batch (missing IDs are "
                "skipped).", len(raw_entries), len(batch),
            )
        for raw in raw_entries[:len(batch)]:
            parsed = _parse_entry(raw, gene_id_field=gene_id_field)
            if parsed is not None:
                entries.append(parsed)

    if gene_id_field != "kegg" and not entries:
        raise ValueError(
            f"None of the KEGG entries carry the gene-id field "
            f"{gene_id_field!r}. Pick a different gene_id_field or "
            f"verify the field name at https://rest.kegg.jp."
        )

    _write_csv(entries, out_path)
    logger.info("Wrote %d KEGG rows to %s", len(entries), out_path)
    return out_path


def _list_genes(sess: requests.Session, kegg_id: str) -> list[str]:
    text = _fetch(sess, _LIST_URL.format(org=kegg_id))
    out: list[str] = []
    prefix = f"{kegg_id}:"
    for line in text.splitlines():
        if not line:
            continue
        first = line.split("\t", 1)[0]
        if first.startswith(prefix):
            out.append(first[len(prefix):])
    return out


def _fetch_get(
    sess: requests.Session, kegg_id: str, batch: list[str]
) -> str:
    query = "+".join(f"{kegg_id}:{g}" for g in batch)
    return _fetch(sess, _GET_URL.format(query=query))


def _fetch(sess: requests.Session, url: str) -> str:
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = sess.get(url, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(_RETRY_BACKOFF * (attempt + 1))
            continue
        # 4xx (other than 429 rate-limiting) means the request is invalid
        # (bad organism code or gene ID); retrying won't help, so fail fast.
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            raise RuntimeError(
                f"KEGG REST returned {resp.status_code} for {url}; "
                f"the requested ID is likely invalid."
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = requests.HTTPError(f"HTTP {resp.status_code}")
            time.sleep(_RETRY_BACKOFF * (attempt + 1))
            continue
        return resp.text
    raise RuntimeError(f"KEGG REST request failed: {url}") from last_err


def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


_ENTRY_RE = re.compile(r"^ENTRY\s+(\S+)", re.MULTILINE)
_UNIPROT_RE = re.compile(r"UniProt:\s*(\S+)")
_EC_RE = re.compile(r"ORTHOLOGY.*?\[EC:([^\]]+)\]", re.DOTALL)
_AASEQ_RE = re.compile(
    r"^AASEQ\s+\d+\s*\n((?:^\s{12,}\S+\s*\n)+)",
    re.MULTILINE,
)
_PATHWAY_RE = re.compile(
    r"^PATHWAY\s+(.*?)(?=^[A-Z]+|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _split_entries(text: str) -> list[str]:
    # KEGG separates entries with "///" on its own line.
    return [e for e in text.split("\n///\n") if e.strip()]


def _parse_entry(
    raw: str, *, gene_id_field: str
) -> dict[str, str] | None:
    # Ensure the entry ends with a newline so line-anchored regexes
    # can match the final field whether or not the splitter trimmed
    # the trailing newline.
    if not raw.endswith("\n"):
        raw = raw + "\n"
    aaseq_match = _AASEQ_RE.search(raw)
    if not aaseq_match:
        return None
    sequence = re.sub(r"\s+", "", aaseq_match.group(1))
    if not sequence:
        return None

    uni_match = _UNIPROT_RE.search(raw)
    if not uni_match:
        return None
    uniprot = uni_match.group(1)

    entry_match = _ENTRY_RE.search(raw)
    if not entry_match:
        return None
    kegg_gene = entry_match.group(1)

    if gene_id_field in ("kegg", ""):
        gene_name = kegg_gene
    else:
        pattern = rf"{re.escape(gene_id_field)}:\s*(\S+)"
        m = re.search(pattern, raw)
        if not m:
            return None
        gene_name = m.group(1)

    ec_match = _EC_RE.search(raw)
    ec = ec_match.group(1).strip() if ec_match else ""

    pathway_match = _PATHWAY_RE.search(raw)
    if pathway_match:
        pathway = re.sub(r"\s+", " ", pathway_match.group(1)).strip()
    else:
        pathway = ""

    mw = str(round(calculate_mw(sequence)))

    return {
        "uniprot": uniprot,
        "gene": gene_name,
        "kegg_gene": kegg_gene,
        "ec": ec,
        "mw": mw,
        "pathway": pathway,
        "sequence": sequence,
    }


def _write_csv(entries: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for e in entries:
            writer.writerow([
                e["uniprot"], e["gene"], e["kegg_gene"], e["ec"],
                e["mw"], e["pathway"], e["sequence"],
            ])
