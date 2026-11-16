"""Download a per-organism UniProt TSV from the UniProt REST API.

Ported from GECKO MATLAB:
src/download_databases/loadDatabases.m
(the UniProt-download branch, extracted into its own module).

Hits ``https://rest.uniprot.org/uniprotkb/stream`` with a query
built from ``id_type`` + ``uniprot_id`` and writes the resulting
TSV (``accession``, ``<gene_id_field>``, ``ec``, ``mass``,
``sequence``) verbatim, matching the schema consumed by
``load_uniprot_tsv``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

_REQUEST_TIMEOUT = 60
_MAX_RETRIES = 5
_RETRY_BACKOFF = 1.0


def download_uniprot(
    uniprot_id: str | int,
    out_path: str | Path,
    *,
    id_type: str = "taxonomy_id",
    gene_id_field: str = "gene_oln",
    reviewed: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download UniProt protein data for an organism and write a TSV.

    Parameters
    ----------
    uniprot_id
        UniProt-side identifier whose meaning depends on ``id_type``.
        For ``id_type='taxonomy_id'`` this is an NCBI taxonomy ID
        (e.g. ``559292`` for *Saccharomyces cerevisiae* S288C).
    out_path
        Destination TSV path (overwritten if it exists).
    id_type
        UniProt query field for ``uniprot_id``. Default
        ``"taxonomy_id"``. MATLAB's adapter accepts the alias
        ``"taxonomy"``; pass either.
    gene_id_field
        UniProt field that becomes the second column (the gene
        matching key). Default ``"gene_oln"`` (Ordered Locus Names)
        works for most prokaryotes and budding yeast.
    reviewed
        If True, restrict the query to reviewed (Swiss-Prot) entries.
        Default False, as the adapter default ``uniprot.reviewed``.
    session
        Optional ``requests.Session`` for connection reuse and test
        injection. A new transient session is used otherwise.

    Returns
    -------
    Path
        The written ``out_path``.

    Raises
    ------
    RuntimeError
        If the UniProt request fails after retries.
    """
    out_path = Path(out_path)
    sess = session or requests.Session()

    # MATLAB accepts 'taxonomy' as an alias for 'taxonomy_id'.
    if id_type == "taxonomy":
        id_type = "taxonomy_id"

    query_parts: list[str] = []
    if reviewed:
        query_parts.append("reviewed:true")
    query_parts.append(f"{id_type}:{uniprot_id}")
    query = " AND ".join(query_parts)

    params = {
        "query": query,
        "fields": f"accession,{gene_id_field},ec,mass,sequence",
        "format": "tsv",
        "compressed": "false",
        "sort": "protein_name asc",
    }

    text = _fetch(sess, _STREAM_URL, params)
    if not text.strip():
        raise RuntimeError(
            f"UniProt returned empty response for query {query!r}. "
            f"Verify the {id_type} value and the gene_id_field."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    n_rows = max(text.count("\n") - 1, 0)
    logger.info("Wrote %d UniProt rows to %s", n_rows, out_path)
    return out_path


def _fetch(
    sess: requests.Session, url: str, params: dict
) -> str:
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = sess.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(_RETRY_BACKOFF * (attempt + 1))
            continue
        # A 4xx (other than 429 rate-limiting) means the request itself is
        # wrong -- a bad query, id_type or gene_id_field. Retrying wastes
        # time and hides the real cause, so surface it immediately with the
        # server's explanation.
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            raise RuntimeError(
                f"UniProt REST request rejected with HTTP "
                f"{resp.status_code}: {resp.text[:200]}"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = requests.HTTPError(f"HTTP {resp.status_code}")
            time.sleep(_RETRY_BACKOFF * (attempt + 1))
            continue
        return resp.text
    raise RuntimeError(
        f"UniProt REST request failed after {_MAX_RETRIES} attempts"
    ) from last_err
