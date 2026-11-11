"""Populate model.ec.eccodes by looking up each reaction's genes in a
protein database.

Ported from GECKO MATLAB:
src/get_enzyme_data/getECfromDatabase.m.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, Optional

from .find_ec_in_db import find_ec_in_db

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..databases import UniprotDB
    from ..databases.kegg_loader import KeggDB
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


@dataclass
class _Conflict:
    """One gene with multiple distinct EC strings in the DB, in the
    context of one reaction. Internal aggregation for the 'display'
    action."""
    rxn_idx: int
    gene: str
    protein_indices: list[int]
    db_name: str = "uniprot"


def fill_eccodes_from_database(
    model: "EcModel",
    uniprot_db: "UniprotDB",
    *,
    kegg_db: Optional["KeggDB"] = None,
    ec_rxns: Optional[Iterable[str]] = None,
    action: Literal["display", "ignore"] = "display",
) -> None:
    """Populate ``model.ec.eccodes`` by looking each reaction's genes
    up in a UniProt database.

    Ported from GECKO MATLAB:
    src/get_enzyme_data/getECfromDatabase.m.

    For each entry of ``model.ec.rxns`` selected by ``ec_rxns``:

    * Gather the gene IDs catalysing the reaction from
      ``model.ec.rxn_enz_mat`` row ``i``.
    * Pass them to ``find_ec_in_db`` along with a pre-built
      gene -> protein-indices map.
    * Store the resulting ``;``-joined EC string in
      ``model.ec.eccodes[i]``.

    The adapter used for gene-ID translation is read from
    ``model.adapter``.

    The gene -> protein-indices map is built once before the loop. It
    tries the adapter's ``uniprotConversion.tsv`` table first: if that
    table changes any gene IDs, each model gene maps to one UniProt ID
    which uniquely identifies one protein. Otherwise it falls back to
    name-based lookup via ``adapter.get_uniprot_compatible_genes``,
    and a single gene may map to several proteins (which is how
    conflicts arise).

    If ``kegg_db`` is supplied, the KEGG fallback fires for any
    reaction whose UniProt-derived EC is empty or ends with ``-``
    (a wildcard). The KEGG hit replaces the UniProt result when
    non-empty.

    Parameters
    ----------
    model
        EcModel with ``model.ec.rxns``, ``model.ec.genes``, and
        ``model.ec.rxn_enz_mat`` already populated (typically by
        ``allocate_ec_for_catalyzed_reactions``,
        ``populate_enzyme_data``, and ``build_rxn_enzyme_coupling``
        respectively). Mutated in place.
    uniprot_db
        Pre-loaded UniProt database. Accessed for its ``eccodes``,
        ``mw``, ``genes``, and ``ids`` fields.
    ec_rxns
        Optional iterable of reaction IDs (each must appear in
        ``model.ec.rxns``) selecting which entries of
        ``model.ec.eccodes`` to update. ``None`` means update all.
        Unknown IDs raise ``ValueError``.
    action
        Behaviour when a gene maps to proteins with multiple distinct
        EC strings. ``'display'`` (default) emits an aggregated
        warning at the end of the run listing every conflicting
        ``(rxn, gene, proteins)`` triple. ``'ignore'`` suppresses the
        aggregated warning. Per-gene warnings from ``find_ec_in_db``
        are emitted in both modes.

    Raises
    ------
    ValueError
        If ``model.adapter`` is None; if ``action`` is not
        ``'display'`` or ``'ignore'``; or if any ID in ``ec_rxns`` is
        not present in ``model.ec.rxns``.
    """
    if action not in ("display", "ignore"):
        raise ValueError(
            f"action must be 'display' or 'ignore', got {action!r}"
        )

    from ..adapter import resolve_adapter
    adapter = resolve_adapter(
        model,
        purpose="fill_eccodes_from_database transforms gene IDs to "
        "UniProt-compatible form via the adapter",
    )

    n = model.ec.n_rxns
    if n == 0:
        return

    if ec_rxns is None:
        positions: list[int] = list(range(n))
    else:
        ec_rxns_list = list(ec_rxns)
        index_by_id = {rid: i for i, rid in enumerate(model.ec.rxns)}
        unknown = [rid for rid in ec_rxns_list if rid not in index_by_id]
        if unknown:
            preview = unknown[:5]
            raise ValueError(
                f"{len(unknown)} reaction ID(s) in ec_rxns are not "
                f"present in model.ec.rxns (examples: {preview})"
            )
        positions = [index_by_id[rid] for rid in ec_rxns_list]

    if not positions:
        return

    gene_to_protein_indices = _build_gene_to_protein_indices(
        model.ec.genes, uniprot_db, adapter,
    )
    kegg_gene_to_indices: dict[str, list[int]] = {}
    if kegg_db is not None:
        kegg_gene_to_indices = _build_kegg_gene_to_indices(
            model.ec.genes, kegg_db, adapter,
        )

    db_eccodes = uniprot_db.eccodes
    db_mw = uniprot_db.mw

    new_eccodes = list(model.ec.eccodes)
    conflicts: list[_Conflict] = []
    rxn_enz_mat = model.ec.rxn_enz_mat.tocsr()

    for i in positions:
        gene_indices = rxn_enz_mat[i].indices
        if len(gene_indices) == 0:
            new_eccodes[i] = ""
            continue
        gene_set = [model.ec.genes[j] for j in gene_indices]

        rxn_conflicts: list[tuple[str, list[int]]] = []
        ec_from_uniprot = find_ec_in_db(
            gene_set, db_eccodes, db_mw, gene_to_protein_indices,
            conflicts=rxn_conflicts,
        )
        chosen_ec = ec_from_uniprot
        chosen_db = "uniprot"

        if kegg_db is not None and (
            not ec_from_uniprot or ec_from_uniprot.endswith("-")
        ):
            kegg_conflicts: list[tuple[str, list[int]]] = []
            ec_from_kegg = find_ec_in_db(
                gene_set, kegg_db.eccodes, kegg_db.mw,
                kegg_gene_to_indices,
                conflicts=kegg_conflicts,
            )
            if ec_from_kegg:
                chosen_ec = ec_from_kegg
                chosen_db = "kegg"
                rxn_conflicts = kegg_conflicts

        new_eccodes[i] = chosen_ec

        for gene, protein_indices in rxn_conflicts:
            conflicts.append(_Conflict(
                rxn_idx=i,
                gene=gene,
                protein_indices=protein_indices,
                db_name=chosen_db,
            ))

    model.ec.eccodes = new_eccodes

    if action == "display" and conflicts:
        logger.warning(_format_conflict_message(conflicts, uniprot_db, kegg_db, model))


def _build_gene_to_protein_indices(
    model_genes: list[str],
    uniprot_db: "UniprotDB",
    adapter: "ModelAdapter",
) -> dict[str, list[int]]:
    """Build the gene -> protein-row-indices map used by ``find_ec_in_db``.

    Tries the conversion table (``uniprotConversion.tsv``) first: if
    that returns different IDs than the gene-name transformation, each
    model gene maps to exactly one UniProt ID -> one protein row.
    Otherwise falls back to gene-name matching, where a single gene
    name may match multiple protein rows.
    """
    if not model_genes:
        return {}

    transformed = adapter.get_uniprot_compatible_genes(model_genes)
    mapped_ids = adapter.get_uniprot_ids_from_table(transformed)

    if mapped_ids != transformed:
        id_to_index = {uid: i for i, uid in enumerate(uniprot_db.ids)}
        result: dict[str, list[int]] = {}
        for original, uid in zip(model_genes, mapped_ids):
            idx = id_to_index.get(uid)
            if idx is not None:
                result[original] = [idx]
        return result

    return _match_genes_by_name(model_genes, transformed, uniprot_db.genes)


def _build_kegg_gene_to_indices(
    model_genes: list[str],
    kegg_db: "KeggDB",
    adapter: "ModelAdapter",
) -> dict[str, list[int]]:
    """Build the gene -> KEGG-row-indices map used by ``find_ec_in_db``."""
    if not model_genes:
        return {}
    transformed = adapter.get_kegg_compatible_genes(model_genes)
    return _match_genes_by_name(model_genes, transformed, kegg_db.genes)


def _match_genes_by_name(
    model_genes: list[str],
    transformed: list[str],
    db_genes: list[str],
) -> dict[str, list[int]]:
    """Map each model gene to the db rows whose gene name equals its
    transformed name (a name may match several rows)."""
    name_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(db_genes):
        if g:
            name_to_indices.setdefault(g, []).append(i)
    result: dict[str, list[int]] = {}
    for original, name in zip(model_genes, transformed):
        indices = name_to_indices.get(name)
        if indices:
            result[original] = list(indices)
    return result


def _format_conflict_message(
    conflicts: list[_Conflict],
    uniprot_db: "UniprotDB",
    kegg_db: Optional["KeggDB"],
    model: "EcModel",
) -> str:
    """Build the aggregated multi-line message for action='display'."""
    rxn_count = len({c.rxn_idx for c in conflicts})
    lines = [
        f"fill_eccodes_from_database: {len(conflicts)} gene-protein conflict(s) "
        f"found across {rxn_count} reaction(s). Resolve by editing your "
        f"UniProt or KEGG data, or call fill_eccodes_from_database with "
        f"action='ignore' to silently use the first match per gene:",
    ]
    for c in conflicts:
        rxn_id = model.ec.rxns[c.rxn_idx]
        if c.db_name == "kegg" and kegg_db is not None:
            protein_ids = [
                kegg_db.uniprot_ids[p] or kegg_db.kegg_genes[p]
                for p in c.protein_indices
            ]
        else:
            protein_ids = [uniprot_db.ids[p] for p in c.protein_indices]
        lines.append(
            f"  - rxn {rxn_id!r} / gene {c.gene!r} ({c.db_name}): "
            f"proteins {', '.join(protein_ids)}"
        )
    return "\n".join(lines)


def get_ec_from_database(
    model: "EcModel",
    uniprot_db: "UniprotDB",
    *,
    kegg_db: Optional["KeggDB"] = None,
    ec_rxns: Optional[Iterable[str]] = None,
    action: Literal["display", "ignore"] = "display",
) -> None:
    """Deprecated alias for :func:`fill_eccodes_from_database`.

    Kept for backward compatibility with the original MATLAB name.
    Will be removed in a future release; switch to
    ``fill_eccodes_from_database``.
    """
    import warnings

    warnings.warn(
        "get_ec_from_database is deprecated; use "
        "fill_eccodes_from_database instead. The old name will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fill_eccodes_from_database(
        model, uniprot_db, kegg_db=kegg_db, ec_rxns=ec_rxns, action=action,
    )
