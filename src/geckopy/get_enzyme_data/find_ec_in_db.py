"""Collect EC codes for a reaction's gene_set from a protein database.

Ported from GECKO MATLAB:
src/geckomat/get_enzyme_data/findECInDB.m.
"""
from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


_EC_SEPARATORS = re.compile(r"[;\s]+")


def _split_ec_cell(cell: str) -> list[str]:
    """Split a database EC cell into its EC tokens.

    UniProt separates codes with ``"; "`` and KEGG with a space, so a cell is
    split on semicolons and whitespace alike and empty pieces are dropped,
    as GECKO MATLAB's ``getECstring`` does.
    """
    return [token for token in _EC_SEPARATORS.split(cell) if token]


def _split_levels(token: str) -> list[str]:
    return token.split(".")


def _subsumes(general: str, specific: str) -> bool:
    """Return True if `general` is a wildcard parent of `specific` (or equal).

    Wildcard semantics matching MATLAB GECKO: a `-` at level N implies
    levels N..end are all wildcards, regardless of what follows in the
    source string. So ``1.-.-.-`` subsumes ``1.2.3.4``, and
    ``1.-.5.5`` is treated identically to ``1.-.-.-``.
    """
    g = _split_levels(general)
    s = _split_levels(specific)
    if len(g) != 4 or len(s) != 4:
        return False
    for i in range(4):
        if g[i] == "-":
            return True
        if s[i] == "-":
            return False
        if g[i] != s[i]:
            return False
    return True


def _wildcard_dedupe(tokens: list[str]) -> list[str]:
    """Drop tokens that are wildcard parents of any other token.

    Preserves insertion order. Equal tokens collapse to the first.
    Mirrors MATLAB GECKO ``compare_wild``.
    """
    n = len(tokens)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            i_in_j = _subsumes(tokens[i], tokens[j])
            j_in_i = _subsumes(tokens[j], tokens[i])
            if i_in_j and j_in_i:
                keep[j] = False
            elif i_in_j:
                keep[i] = False
                break
            elif j_in_i:
                keep[j] = False
    return [t for k, t in zip(keep, tokens) if k]


def _wildcard_intersection(prev: list[str], new: list[str]) -> list[str]:
    """Pairwise wildcard-aware intersection of two token lists.

    For each pair (a, b), if one subsumes the other, the more specific
    one (or either, when equal) is added. Disjoint pairs contribute
    nothing. The result is not deduplicated here; duplicates are
    cleaned up separately by the caller via ``_wildcard_dedupe``.
    """
    result: list[str] = []
    for a in prev:
        for b in new:
            survivors = _wildcard_dedupe([a, b])
            if len(survivors) == 1:
                result.append(survivors[0])
    return result


def find_ec_in_db(
    gene_set: list[str],
    db_eccodes: list[str],
    db_mw: np.ndarray,
    gene_to_protein_indices: dict[str, list[int]],
    *,
    conflicts: list[tuple[str, list[int]]] | None = None,
) -> str:
    """Collect EC codes for a reaction's genes from a protein database.

    Ported from GECKO MATLAB:
    src/geckomat/get_enzyme_data/findECInDB.m.

    Per-gene resolution:

    * Look up the gene's protein indices in
      ``gene_to_protein_indices``. Missing keys count as no DB match.
    * Filter to proteins with non-empty ``db_eccodes``.
    * If those proteins agree on a single EC string (or there is only
      one protein), pick the lightest-MW one's string. NaN MWs are
      treated as +inf so a NaN-only group falls back to the first.
    * If they disagree, pick the first distinct string in DB order
      (matching MATLAB ``unique('stable')``) and emit a
      ``logger.warning`` listing the gene as a conflict.

    Cross-gene reconciliation (for complexes with multiple subunits):

    * UNION of all gene-level token sets, with wildcard subsumption
      (``1.1.1.1`` removes ``1.1.1.-``).
    * INTERSECTION over the same, pairwise.
    * Use intersection if non-empty, else union. Final result is
      wildcard-deduped and joined with ``;``.

    Parameters
    ----------
    gene_set
        Gene IDs from one reaction's GPR (one AND-clause after
        isozyme expansion). Order matters for reproducibility of the
        cross-gene UNION / INTERSECTION ordering.
    db_eccodes
        EC string per protein row in the database. Empty string means
        "no EC for this protein". A cell may hold several codes, separated
        by semicolons, spaces or both (``"1.1.1.1; 2.2.2.2"`` from UniProt,
        ``"1.1.1.4 1.1.1.-"`` from KEGG).
    db_mw
        Molecular weight per protein row, in Da. NaN is tolerated.
    gene_to_protein_indices
        Maps gene ID to a list of protein row indices in the DB. Built
        once by ``getECfromDatabase`` before iterating reactions.
        Missing keys mean "this gene has no DB match" (silent).
    conflicts
        Optional accumulator. If provided, ``(gene,
        protein_indices)`` tuples are appended for every gene in
        ``gene_set`` that maps to multiple distinct EC strings, where
        ``protein_indices`` are the DB row indices of the first
        protein per distinct EC (mirroring MATLAB's
        ``unique('stable')`` indexing). Useful for aggregating
        conflicts across many calls without scraping logs.

    Returns
    -------
    str
        ``;``-joined EC tokens (no ``EC`` prefix), e.g.
        ``"1.1.1.1;2.7.7.7"``. Empty string if no gene contributed
        any EC.
    """
    per_gene_tokens: list[list[str]] = []

    for gene in gene_set:
        protein_indices = gene_to_protein_indices.get(gene, [])
        matched = [p for p in protein_indices if db_eccodes[p]]
        if not matched:
            per_gene_tokens.append([])
            continue

        # Distinct EC strings, preserving DB order. Track the first
        # protein index seen per distinct EC so callers can collect
        # them as conflict evidence.
        seen: set[str] = set()
        distinct: list[str] = []
        distinct_proteins: list[int] = []
        for p in matched:
            ec = db_eccodes[p]
            if ec not in seen:
                seen.add(ec)
                distinct.append(ec)
                distinct_proteins.append(p)

        if len(distinct) > 1:
            logger.warning(
                "find_ec_in_db: gene %r maps to %d distinct EC string(s) "
                "in the database: %s. Using first: %r.",
                gene,
                len(distinct),
                distinct,
                distinct[0],
            )
            chosen = distinct[0]
            if conflicts is not None:
                conflicts.append((gene, distinct_proteins))
        else:
            mws = db_mw[matched]
            finite_mws = np.where(np.isnan(mws), np.inf, mws)
            min_idx = int(np.argmin(finite_mws))
            chosen = db_eccodes[matched[min_idx]]

        per_gene_tokens.append(_split_ec_cell(chosen))

    per_gene_tokens = [t for t in per_gene_tokens if t]

    if not per_gene_tokens:
        return ""

    if len(per_gene_tokens) == 1:
        return ";".join(_wildcard_dedupe(per_gene_tokens[0]))

    union_tokens = list(per_gene_tokens[0])
    intersection_tokens = list(per_gene_tokens[0])
    for next_tokens in per_gene_tokens[1:]:
        union_tokens = _wildcard_dedupe(union_tokens + next_tokens)
        intersection_tokens = _wildcard_intersection(
            intersection_tokens, next_tokens
        )

    result_tokens = intersection_tokens if intersection_tokens else union_tokens
    result_tokens = _wildcard_dedupe(result_tokens)

    return ";".join(result_tokens)
