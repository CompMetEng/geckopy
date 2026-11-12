"""Build and populate the ec substructure on an EcModel.

Covers stages 6, 7, and 8 of GECKO MATLAB `makeEcModel.m`:

- Stage 6 allocates empty per-reaction slots in ec (one per catalyzed
  reaction after stage-5 expansion).
- Stage 7 fills in per-enzyme data (genes, uniprot IDs, MW, sequence,
  concentrations) by looking up each model gene in the UniProt
  database. Genes that do not match UniProt are left out of ec and
  returned as a warning list; their parent reactions are also flagged
  with a note in rxn.notes.
- Stage 8 populates the reaction-to-enzyme coupling matrix
  (ec.rxn_enz_mat) with 1.0 for each (reaction, enzyme) pair implied
  by the reaction's GPR.

These three stages are only meaningful for the full (non-light) ecModel
format. Gecko-light is handled by a separate future module.

All three functions mutate the ec substructure in place. None of them
talk to the network.

Ported from GECKO MATLAB: src/change_model/makeEcModel.m.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from ...databases import UniprotDB
    from ...databases.kegg_loader import KeggDB
    from ..ec_model import EcModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage 6: allocate empty ec slots
# --------------------------------------------------------------------------- #

def allocate_ec_for_catalyzed_reactions(model: "EcModel") -> list[str]:
    """Stage 6: allocate ec.rxns and per-reaction slots.

    Applies only to full (non-light) ecModels; gecko-light uses
    :func:`allocate_ec_and_coupling_light` instead.

    Walks the model (which must already have been through stages 1-5)
    and selects every reaction that has at least one gene associated.
    Each selected reaction gets an entry in ec.rxns; ec.kcat is
    initialized to 0 (0 marks "no kcat assigned"), and
    ec.source/notes/eccodes to empty strings. The per-enzyme arrays
    (including ec.concs) are left untouched (stage 7 populates them).

    Parameters
    ----------
    model
        An EcModel, already preprocessed through stage 5. Mutated in place.

    Returns
    -------
    list of str
        The reaction IDs written into ec.rxns, in model order.
    """
    rxn_ids: list[str] = [r.id for r in model.reactions if r.genes]
    n = len(rxn_ids)

    model.ec.rxns = rxn_ids
    # 0 marks "no kcat assigned" (matching MATLAB GECKO).
    model.ec.kcat = np.zeros(n, dtype=float)
    model.ec.source = [""] * n
    model.ec.notes = [""] * n
    model.ec.eccodes = [""] * n
    # ec.concs is a per-enzyme field, not per-reaction. Stage 7 sizes it.

    return rxn_ids


# --------------------------------------------------------------------------- #
# Stage 7: populate per-enzyme data from UniProt
# --------------------------------------------------------------------------- #

_UNCONSTRAINED_NOTE_KEY = "geckopy_warning"


def populate_enzyme_data(
    model: "EcModel",
    uniprot_db: "UniprotDB",
    *,
    kegg_db: "KeggDB | None" = None,
) -> list[str]:
    """Stage 7: fill per-enzyme fields from UniProt (with optional KEGG fallback).

    For every gene in the model (sorted alphabetically for reproducibility),
    a lookup is performed:

    1. ``adapter.get_uniprot_compatible_genes`` transforms model gene IDs
       into a form that should match the UniProt "Gene Names" column.
    2. If an ``uniprotConversion.tsv`` is present, it instead transforms
       model gene IDs directly to UniProt IDs and matches on the "Entry"
       column.

    Matched genes are added to ec.genes / ec.enzymes / ec.mw /
    ec.sequence (same length as ec.genes), and ec.concs is allocated as
    NaN of matching length. Unmatched genes are returned as a list.

    When ``kegg_db`` is supplied, each gene UniProt could not match is
    looked up in the KEGG database via
    ``adapter.get_kegg_compatible_genes``. KEGG hits feed ec.enzymes
    with the UniProt accession stored on the KEGG entry; if that is
    empty, the bare KEGG gene ID (``KeggDB.kegg_genes``) is used as a
    fallback identifier. The returned list then names only the genes
    that neither source could fill.

    Reactions whose GPR references an unmatched gene get a warning note
    added to ``rxn.notes[geckopy_warning]``. If such a reaction has at
    least one other matched gene, it will still be enzyme-constrained
    through that gene. If all its genes are unmatched it will be
    unconstrained.

    Parameters
    ----------
    model
        An EcModel. Must have ``adapter`` set. Mutated in place.
    uniprot_db
        Loaded UniProt database.
    kegg_db
        Optional pre-loaded KEGG database, consulted as a fallback for
        genes UniProt does not match. ``None`` (default) skips the
        KEGG lookup entirely.

    Returns
    -------
    list of str
        Model gene IDs (pre-transformation) that did not match.
    """
    from ...adapter import resolve_adapter
    adapter = resolve_adapter(
        model,
        purpose="populate_enzyme_data transforms gene IDs to "
        "UniProt-compatible form via the adapter",
    )

    # Sorted alphabetically so downstream indexing is deterministic.
    model_genes: list[str] = sorted(g.id for g in model.genes)
    if not model_genes:
        model.ec.genes = []
        model.ec.enzymes = []
        model.ec.mw = np.empty(0, dtype=float)
        model.ec.sequence = []
        model.ec.concs = np.empty(0, dtype=float)
        return []

    # Step 1: try gene-based lookup via adapter.get_uniprot_compatible_genes
    transformed = adapter.get_uniprot_compatible_genes(model_genes)
    matches: list[int | None] = [
        uniprot_db.find_by_gene(g) for g in transformed
    ]

    # Step 2: if the adapter has a conversion table, that takes over.
    mapped_ids = adapter.get_uniprot_ids_from_table(transformed)
    if mapped_ids != transformed:
        matches = [
            uniprot_db.find_by_id(i) if i else None for i in mapped_ids
        ]

    matched_genes: list[str] = []
    matched_enzymes: list[str] = []
    matched_mw: list[float] = []
    matched_seq: list[str] = []
    unmatched: list[str] = []
    kegg_filled: list[str] = []
    kegg_used_bare_id: list[tuple[str, str]] = []

    kegg_lookup: dict[str, int] = {}
    kegg_transformed: list[str] = []
    if kegg_db is not None and len(kegg_db) > 0:
        kegg_transformed = adapter.get_kegg_compatible_genes(model_genes)
        for i, g in enumerate(kegg_db.genes):
            if g and g not in kegg_lookup:
                kegg_lookup[g] = i

    for idx, (gene, row_idx) in enumerate(zip(model_genes, matches)):
        if row_idx is not None:
            matched_genes.append(gene)
            matched_enzymes.append(uniprot_db.ids[row_idx])
            matched_mw.append(uniprot_db.mw[row_idx])
            matched_seq.append(uniprot_db.sequences[row_idx])
            continue
        kegg_idx: int | None = None
        if kegg_lookup:
            kegg_idx = kegg_lookup.get(kegg_transformed[idx])
        if kegg_idx is None:
            unmatched.append(gene)
            continue
        # KEGG fallback hit. Prefer the UniProt accession given in the
        # KEGG row; fall back to the bare KEGG gene id when empty.
        kegg_uniprot = kegg_db.uniprot_ids[kegg_idx]
        if kegg_uniprot:
            enzyme_id = kegg_uniprot
        else:
            enzyme_id = kegg_db.kegg_genes[kegg_idx]
            kegg_used_bare_id.append((gene, enzyme_id))
        matched_genes.append(gene)
        matched_enzymes.append(enzyme_id)
        matched_mw.append(float(kegg_db.mw[kegg_idx]))
        matched_seq.append(kegg_db.sequences[kegg_idx])
        kegg_filled.append(gene)

    if not matched_genes:
        raise ValueError(
            "None of the model genes matched an entry in the UniProt "
            "database (or KEGG, if provided). Check "
            "adapter.params.uniprot.* / adapter.params.kegg.* and, "
            "if needed, provide a data/uniprotConversion.tsv file."
        )

    model.ec.genes = matched_genes
    model.ec.enzymes = matched_enzymes
    model.ec.mw = np.array(matched_mw, dtype=float)
    model.ec.sequence = matched_seq
    model.ec.concs = np.full(len(matched_genes), np.nan, dtype=float)

    # Per-enzyme rows are keyed by gene, so two genes that map to the same
    # UniProt accession produce duplicate ec.enzymes entries. apply_kcat now
    # sums their coupling, but the duplication is worth surfacing because the
    # per-enzyme mw/concs of the second row are otherwise easy to overlook.
    seen: set[str] = set()
    duplicates = sorted({e for e in matched_enzymes if e in seen or seen.add(e)})
    if duplicates:
        logger.warning(
            "populate_enzyme_data: %d UniProt accession(s) are shared by "
            "multiple genes (e.g. %s); their enzyme rows are duplicated in "
            "model.ec.enzymes.",
            len(duplicates), ", ".join(duplicates[:5]),
        )

    if kegg_filled:
        logger.info(
            "populate_enzyme_data: %d gene(s) resolved via KEGG fallback "
            "(UniProt had no entry): %s",
            len(kegg_filled),
            ", ".join(kegg_filled[:10]) + ("..." if len(kegg_filled) > 10 else ""),
        )
        if kegg_used_bare_id:
            preview = ", ".join(
                f"{g}->{eid}" for g, eid in kegg_used_bare_id[:10]
            )
            more = (
                "" if len(kegg_used_bare_id) <= 10
                else f" (and {len(kegg_used_bare_id) - 10} more)"
            )
            logger.warning(
                "populate_enzyme_data: %d KEGG-resolved gene(s) had no "
                "UniProt accession on the KEGG row; the bare KEGG gene id "
                "is used in ec.enzymes instead: %s%s",
                len(kegg_used_bare_id), preview, more,
            )

    # Annotate reactions whose GPR mentions any unmatched gene.
    if unmatched:
        unmatched_set = set(unmatched)
        for rxn in model.reactions:
            if not rxn.genes:
                continue
            missing = sorted(
                g.id for g in rxn.genes if g.id in unmatched_set
            )
            if missing:
                rxn.notes[_UNCONSTRAINED_NOTE_KEY] = (
                    "geckopy: no ec-constraint due to no uniprot/kegg match "
                    "for gene(s): " + ", ".join(missing)
                )

    return unmatched


# --------------------------------------------------------------------------- #
# Stage 8: build ec.rxn_enz_mat
# --------------------------------------------------------------------------- #

def build_rxn_enzyme_coupling(model: "EcModel") -> None:
    """Stage 8: populate ec.rxn_enz_mat.

    For each reaction ID in ec.rxns and each gene in ec.genes, writes
    1.0 at (i, j) if the reaction's GPR contains that gene. After
    stage 5, every reaction has at most one AND-clause, so this is
    simply "which enzymes are subunits of the complex catalyzing this
    reaction."

    Downstream functions (e.g. applyComplexData) may later overwrite
    these 1.0 entries with actual subunit stoichiometries.

    Parameters
    ----------
    model
        An EcModel with ec.rxns and ec.genes already populated
        (stages 6 and 7). Mutated in place.
    """
    n_rxns = model.ec.n_rxns
    n_enz = model.ec.n_enzymes

    if n_rxns == 0 or n_enz == 0:
        model.ec.rxn_enz_mat = sparse.csr_matrix((n_rxns, n_enz), dtype=float)
        return

    enzyme_index: dict[str, int] = {g: i for i, g in enumerate(model.ec.genes)}

    mat = sparse.lil_matrix((n_rxns, n_enz), dtype=float)
    for i, rxn_id in enumerate(model.ec.rxns):
        rxn = model.reactions.get_by_id(rxn_id)
        for gene in rxn.genes:
            j = enzyme_index.get(gene.id)
            if j is not None:
                mat[i, j] = 1.0

    model.ec.rxn_enz_mat = mat.tocsr()


# --------------------------------------------------------------------------- #
# gecko-light replacement for stages 6 + 8
# --------------------------------------------------------------------------- #

# Width of the `###_` counter prefix that distinguishes isozyme copies of the
# same cobra reaction in ec.rxns under the gecko-light layout. Three digits
# accommodate up to 999 isozymes per reaction, matching MATLAB GECKO.
_LIGHT_PREFIX_WIDTH = 3


def allocate_ec_and_coupling_light(model: "EcModel") -> list[str]:
    """Stages 6 + 8 (gecko-light layout): allocate ec.rxns and ec.rxn_enz_mat.

    Light models do not split isozyme reactions in the cobra layer
    (expand_model is skipped), so the per-isozyme bookkeeping moves
    into ec instead:

    - Each cobra reaction with N isozymes (i.e. ``len(gpr_to_dnf(rxn.gpr))
      == N``) produces N rows in ec.rxns. Row k has the id
      ``"{k:03d}_{rxn.id}"`` (1-indexed, three-digit zero-padded counter).
    - Row k's slice of ``ec.rxn_enz_mat`` is 1.0 for each gene in that
      isozyme's AND-clause; other genes are 0.

    ``ec.kcat`` is initialized to 0 ("no kcat assigned"), and
    ``ec.source`` / ``ec.notes`` / ``ec.eccodes`` to empty strings — same
    convention as :func:`allocate_ec_for_catalyzed_reactions`.

    Stage 7 (``populate_enzyme_data``) runs unchanged on light models, so
    ``ec.genes`` / ``ec.enzymes`` / ``ec.mw`` / ``ec.sequence`` must
    already be populated before this function builds the coupling matrix.
    Call order in :func:`make_ec_model` for light: ``populate_enzyme_data``
    first, then this function.

    Parameters
    ----------
    model
        An EcModel preprocessed through stage 4 (irreversibility), with
        stage 5 (expand_model) skipped and stage 7 already complete.
        Mutated in place.

    Returns
    -------
    list of str
        The prefixed reaction IDs written into ``model.ec.rxns``.
    """
    from raven_toolbox.manipulation.expand import gpr_to_dnf

    enzyme_index: dict[str, int] = {
        g: i for i, g in enumerate(model.ec.genes)
    }
    n_enz = model.ec.n_enzymes

    rxn_ids: list[str] = []
    # Buffer (row, gene_index) pairs and build the sparse matrix in one pass
    # at the end; faster than incrementally mutating an lil_matrix per row.
    row_for_gene: list[list[int]] = []

    for rxn in model.reactions:
        if not rxn.genes:
            continue
        clauses = gpr_to_dnf(rxn.gpr)
        if not clauses:
            continue
        for k, clause in enumerate(clauses, start=1):
            rxn_ids.append(f"{k:0{_LIGHT_PREFIX_WIDTH}d}_{rxn.id}")
            row_for_gene.append([
                enzyme_index[g]
                for g in clause
                if g in enzyme_index
            ])

    n_rxns = len(rxn_ids)

    model.ec.rxns = rxn_ids
    # 0 marks "no kcat assigned" (matching MATLAB GECKO).
    model.ec.kcat = np.zeros(n_rxns, dtype=float)
    model.ec.source = [""] * n_rxns
    model.ec.notes = [""] * n_rxns
    model.ec.eccodes = [""] * n_rxns

    if n_rxns == 0 or n_enz == 0:
        model.ec.rxn_enz_mat = sparse.csr_matrix(
            (n_rxns, n_enz), dtype=float,
        )
        return rxn_ids

    mat = sparse.lil_matrix((n_rxns, n_enz), dtype=float)
    for i, gene_indices in enumerate(row_for_gene):
        for j in gene_indices:
            mat[i, j] = 1.0
    model.ec.rxn_enz_mat = mat.tocsr()
    return rxn_ids


def split_light_rxn_id(prefixed_id: str) -> tuple[int, str]:
    """Split a gecko-light prefixed ec.rxn id into ``(isozyme_index, cobra_id)``.

    Light models prefix each ec.rxns entry with a fixed-width zero-padded
    counter (e.g. ``"001_RXN1"`` -> ``(1, "RXN1")``). Helpers that need
    to look up the underlying cobra reaction use this to strip the prefix.
    Raises ``ValueError`` on an unprefixed id.
    """
    if (
        len(prefixed_id) <= _LIGHT_PREFIX_WIDTH + 1
        or prefixed_id[_LIGHT_PREFIX_WIDTH] != "_"
        or not prefixed_id[:_LIGHT_PREFIX_WIDTH].isdigit()
    ):
        raise ValueError(
            f"{prefixed_id!r} is not a gecko-light ec.rxns id "
            f"(expected '<3-digit>_<cobra_id>')"
        )
    return int(prefixed_id[:_LIGHT_PREFIX_WIDTH]), prefixed_id[_LIGHT_PREFIX_WIDTH + 1:]
