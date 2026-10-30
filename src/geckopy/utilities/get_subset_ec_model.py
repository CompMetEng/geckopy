"""Extract a context-specific ecModel as a subset of a bigger ecModel.

Ported from GECKO MATLAB:
src/geckomat/utilities/getSubsetEcModel.m.
"""
from __future__ import annotations

import contextlib
import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
from cobra.manipulation import remove_genes

from ..solver_support import edit_on_fast_solver
from ..ec_model.constants import (
    POOL_EXCHANGE_ID,
    PROT_PREFIX,
    USAGE_PREFIX,
    canonicalize_rxn_id,
)

if TYPE_CHECKING:
    import cobra

    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)


_STANDARD_GENE = "standard"
_USAGE_DEFAULT_UB = 1000.0


def get_subset_ec_model(
    big_ec_model: "EcModel",
    small_gem: "cobra.Model",
) -> "EcModel":
    """Trim ``big_ec_model`` to the gene/reaction set of ``small_gem``.

    Both models must have been derived from the same starting GEM.
    Reactions in ``small_gem`` are matched against canonicalised
    (suffix-stripped) reaction ids in ``big_ec_model``; if any
    ``small_gem`` reaction is missing, the function raises.

    Returns a fresh `EcModel`; the input ``big_ec_model`` is not
    mutated.

    Algorithm (mirrors MATLAB):

    1. Copy ``big_ec_model``.
    2. Compute genes to remove (`big.genes` minus `small_gem.genes`,
       preserving the ``standard`` pseudo-gene from
       ``assign_standard_kcat``).
    3. Remove the genes with ``cobra.manipulation.remove_genes``: the
       reactions the removal blocks go, and the genes leave the rule of
       every reaction that survives (an isozyme remains).
    4. Trim the corresponding columns of
       ``model.ec.genes/enzymes/mw/sequence/concs/rxn_enz_mat``.
    5. Remove reactions whose canonicalised id is not in
       ``small_gem.reactions``, except ``usage_prot_*`` and
       ``prot_pool_exchange`` which are always preserved.
    6. Remove the metabolites and genes that no reaction uses, as
       MATLAB's ``removeGenes`` and ``removeReactions`` do.
    7. Trim the corresponding rows of
       ``model.ec.rxns/kcat/source/notes/eccodes/rxn_enz_mat``. In a
       gecko-light model the ec reaction ids carry a ``001_`` prefix that
       is not part of the reaction id.

    Ported from GECKO MATLAB:
    src/geckomat/utilities/getSubsetEcModel.m.

    If any ``usage_prot_*`` reaction's upper bound differs from the
    default (1000), a warning is logged: protein-concentration
    constraints on ``big_ec_model`` may not carry over meaningfully
    to the subset.

    Parameters
    ----------
    big_ec_model
        The full ecModel covering many genes/reactions.
    small_gem
        The conventional (non-ec) context-specific model whose
        gene/reaction set is the target.

    Returns
    -------
    EcModel
        A new ecModel restricted to `small_gem`'s gene/reaction set,
        with the protein pool / usage rxn machinery preserved.

    Raises
    ------
    ValueError
        If any reaction in ``small_gem`` is missing from
        ``big_ec_model`` (after suffix stripping).
    """
    # --- 1. Validate small_gem reactions are a subset of big's ---
    big_canonical_ids = {
        _canonical(r.id) for r in big_ec_model.reactions
    }
    missing = [
        r.id for r in small_gem.reactions
        if r.id not in big_canonical_ids
    ]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} reaction(s) in small_gem not found in "
            f"big_ec_model after suffix stripping (examples: "
            f"{preview}). Were both models derived from the same "
            f"starting GEM?"
        )

    # --- 2. Warn on context-dependent protein constraints ---
    for rxn in big_ec_model.reactions:
        if rxn.id.startswith(USAGE_PREFIX):
            if rxn.upper_bound != _USAGE_DEFAULT_UB:
                logger.warning(
                    "get_subset_ec_model: big_ec_model has protein-"
                    "concentration constraints (e.g. %s.upper_bound = %g) "
                    "that may not be relevant in the subset model.",
                    rxn.id, rxn.upper_bound,
                )
                break

    # --- 3. Copy the model ---
    # cobra.Model.copy() doesn't know about the `ec` substructure, so
    # deep-copy it explicitly to avoid mutating the input model.
    small = big_ec_model.copy()
    small.ec = copy.deepcopy(big_ec_model.ec)
    # The removals below run on glpk when the model is on scipy and the model is
    # converted back at the end (the scipy interface rewrites every constraint
    # expression for a removed variable).
    solver_edit = contextlib.ExitStack()
    solver_edit.enter_context(edit_on_fast_solver(small))

    # --- 4. Compute genes to remove ---
    small_gem_gene_ids = {g.id for g in small_gem.genes}
    big_gene_ids = {g.id for g in small.genes}
    genes_to_remove = (
        big_gene_ids - small_gem_gene_ids - {_STANDARD_GENE}
    )

    # --- 5. Remove the genes, and the reactions they block ---
    # A reaction that survives (an isozyme remains) loses the gene from its
    # rule, so no kept reaction refers to a gene that is no longer in
    # model.genes. A complex that loses a subunit is dropped whole.
    if genes_to_remove:
        remove_genes(small, sorted(genes_to_remove), remove_reactions=True)

    # --- 7. Trim ec per-enzyme fields ---
    if genes_to_remove:
        old_enzymes = list(small.ec.enzymes)
        keep_enz_mask = np.array(
            [g not in genes_to_remove for g in small.ec.genes], dtype=bool,
        )
        small.ec.genes = [
            g for g, k in zip(small.ec.genes, keep_enz_mask) if k
        ]
        small.ec.enzymes = [
            e for e, k in zip(small.ec.enzymes, keep_enz_mask) if k
        ]
        small.ec.mw = small.ec.mw[keep_enz_mask]
        small.ec.sequence = [
            s for s, k in zip(small.ec.sequence, keep_enz_mask) if k
        ]
        if small.ec.concs.size == keep_enz_mask.size:
            small.ec.concs = small.ec.concs[keep_enz_mask]
        if small.ec.rxn_enz_mat.shape[1] == keep_enz_mask.size:
            small.ec.rxn_enz_mat = small.ec.rxn_enz_mat[
                :, keep_enz_mask
            ].tocsr()

        # Drop the protein-pool machinery (usage_prot_<X> reaction and
        # prot_<X> metabolite) for accessions no longer used by any kept
        # enzyme row, so the subset carries no orphaned enzyme constraints.
        # An accession shared by a kept gene (duplicate accession) is
        # retained.
        kept_accessions = set(small.ec.enzymes)
        orphaned = [
            e for e, k in zip(old_enzymes, keep_enz_mask)
            if not k and e not in kept_accessions
        ]
        orphan_usage = [
            small.reactions.get_by_id(f"{USAGE_PREFIX}{e}")
            for e in orphaned
            if small.reactions.has_id(f"{USAGE_PREFIX}{e}")
        ]
        if orphan_usage:
            small.remove_reactions(orphan_usage, remove_orphans=False)
        orphan_mets = [
            small.metabolites.get_by_id(f"{PROT_PREFIX}{e}")
            for e in orphaned
            if small.metabolites.has_id(f"{PROT_PREFIX}{e}")
        ]
        if orphan_mets:
            small.remove_metabolites(orphan_mets)

    # --- 8. Remove reactions whose canonical id is not in small_gem ---
    small_gem_rxn_ids = {r.id for r in small_gem.reactions}
    rxns_to_remove_for_subset = []
    for rxn in list(small.reactions):
        if rxn.id.startswith(USAGE_PREFIX) or rxn.id == POOL_EXCHANGE_ID:
            continue
        if _canonical(rxn.id) not in small_gem_rxn_ids:
            rxns_to_remove_for_subset.append(rxn)
    if rxns_to_remove_for_subset:
        small.remove_reactions(
            rxns_to_remove_for_subset, remove_orphans=True,
        )

    # --- 8b. Remove the metabolites and genes no reaction uses ---
    # getSubsetEcModel's removeGenes and removeReactions calls drop every
    # metabolite without a reaction and every gene without a reaction, also
    # those that were unused in the big model (such as the ``standard``
    # pseudo-gene of a light model).
    unused_mets = [m for m in small.metabolites if not m.reactions]
    if unused_mets:
        small.remove_metabolites(unused_mets, destructive=False)
    unused_genes = [g for g in small.genes if not g.reactions]
    if unused_genes:
        remove_genes(small, unused_genes, remove_reactions=False)

    solver_edit.close()

    # --- 9. Trim ec per-rxn fields ---
    cobra_rxn_ids = {r.id for r in small.reactions}
    keep_rxn_mask = np.array(
        [
            (rid[4:] if small.ec.gecko_light else rid) in cobra_rxn_ids
            for rid in small.ec.rxns
        ],
        dtype=bool,
    )
    if not keep_rxn_mask.all():
        small.ec.rxns = [
            r for r, k in zip(small.ec.rxns, keep_rxn_mask) if k
        ]
        small.ec.kcat = small.ec.kcat[keep_rxn_mask]
        small.ec.source = [
            s for s, k in zip(small.ec.source, keep_rxn_mask) if k
        ]
        small.ec.notes = [
            n for n, k in zip(small.ec.notes, keep_rxn_mask) if k
        ]
        small.ec.eccodes = [
            c for c, k in zip(small.ec.eccodes, keep_rxn_mask) if k
        ]
        if small.ec.rxn_enz_mat.shape[0] == keep_rxn_mask.size:
            small.ec.rxn_enz_mat = small.ec.rxn_enz_mat[
                keep_rxn_mask, :
            ].tocsr()

    # Surface any length/shape drift introduced by the trims rather than
    # returning a silently inconsistent ec substructure.
    small.ec.validate()
    return small


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _canonical(rxn_id: str) -> str:
    """Strip ``_REV`` and ``_EXP_<N>`` suffixes."""
    return canonicalize_rxn_id(rxn_id)[0]
