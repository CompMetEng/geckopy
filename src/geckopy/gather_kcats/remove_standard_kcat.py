"""Remove the "standard" pseudoenzyme and standard kcat assignments
added by `assign_standard_kcat`.

Ported from GECKO MATLAB:
src/gather_kcats/removeStandardKcat.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..ec_model.pipeline.apply_kcat import apply_kcat_constraints

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


_STANDARD_NAME = "standard"
_STANDARD_MET_ID = "prot_standard"
_STANDARD_USAGE_RXN_ID = "usage_prot_standard"


def remove_standard_kcat(model: "EcModel") -> None:
    """Remove the standard pseudoenzyme and any standard kcat
    assignments from ``model``.

    Low-level: it undoes ``assign_standard_kcat``, which ``collect_kcats``
    and ``assign_kcats`` run; call it directly only to change kcats by hand.

    Ported from GECKO MATLAB:
    src/gather_kcats/removeStandardKcat.m.

    Inverse of ``assign_standard_kcat``. Cleans up:

    1. Reactions in ``ec.rxns`` whose row in ``rxn_enz_mat`` points to
       the "standard" pseudoenzyme are dropped from all per-reaction
       ec fields.
    2. The "standard" entry is dropped from all per-enzyme ec fields,
       and its column is removed from ``rxn_enz_mat``.
    3. Remaining ec.rxns rows where ``source == "standard"`` (these
       are rxns whose existing kcat was filled by
       ``fill_zero_kcat=True``, but linked to a real enzyme) have
       their ``kcat`` reset to 0 and ``source`` cleared.
    4. ``apply_kcat_constraints`` is invoked on the affected rxns to
       update the ``S`` matrix.
    5. ``prot_standard`` metabolite, ``usage_prot_standard``
       reaction, and the ``"standard"`` gene are removed from the
       cobra model (if present).

    Idempotent: running on a model with no standard pseudoenzyme is
    a no-op.

    Parameters
    ----------
    model
        EcModel with a standard pseudoenzyme installed by
        ``assign_standard_kcat``. Mutated in place.
    """
    _remove_standard_pseudoenzyme(model)
    affected = _reset_standard_source_rxns(model)
    if affected:
        apply_kcat_constraints(model, update_rxns=affected)
    _remove_standard_topology(model)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _remove_standard_pseudoenzyme(model: "EcModel") -> None:
    """Drop the 'standard' pseudoenzyme entry and any ec.rxns rows
    linked to it via rxn_enz_mat."""
    if _STANDARD_NAME not in model.ec.enzymes:
        return

    std_col = model.ec.enzymes.index(_STANDARD_NAME)

    # Identify rxn_enz_mat rows that point to the standard column.
    if model.ec.rxn_enz_mat.shape[0] > 0:
        std_col_dense = (
            model.ec.rxn_enz_mat[:, std_col].toarray().ravel()
        )
        rows_to_drop_mask = std_col_dense > 0
    else:
        rows_to_drop_mask = np.zeros(0, dtype=bool)

    if rows_to_drop_mask.any():
        keep = ~rows_to_drop_mask
        model.ec.rxns = [r for r, k in zip(model.ec.rxns, keep) if k]
        model.ec.kcat = model.ec.kcat[keep]
        model.ec.source = [s for s, k in zip(model.ec.source, keep) if k]
        model.ec.notes = [n for n, k in zip(model.ec.notes, keep) if k]
        model.ec.eccodes = [c for c, k in zip(model.ec.eccodes, keep) if k]
        model.ec.rxn_enz_mat = model.ec.rxn_enz_mat[keep, :].tocsr()

    # Drop the column for the standard enzyme.
    n_enz = model.ec.n_enzymes
    keep_cols = np.ones(n_enz, dtype=bool)
    keep_cols[std_col] = False
    model.ec.rxn_enz_mat = model.ec.rxn_enz_mat[:, keep_cols].tocsr()

    # Drop the per-enzyme entries.
    model.ec.genes = [g for g, k in zip(model.ec.genes, keep_cols) if k]
    model.ec.enzymes = [e for e, k in zip(model.ec.enzymes, keep_cols) if k]
    model.ec.mw = model.ec.mw[keep_cols]
    model.ec.sequence = [
        s for s, k in zip(model.ec.sequence, keep_cols) if k
    ]
    if model.ec.concs.size == n_enz:
        model.ec.concs = model.ec.concs[keep_cols]


def _reset_standard_source_rxns(model: "EcModel") -> list[str]:
    """Reset rows where source='standard' to kcat=0 and source=''.

    Returns the affected reaction IDs so the caller can re-apply kcat
    constraints to clean up the S matrix.
    """
    affected: list[str] = []
    for i, src in enumerate(model.ec.source):
        if src == "standard":
            model.ec.kcat[i] = 0.0
            model.ec.source[i] = ""
            affected.append(model.ec.rxns[i])
    return affected


def _remove_standard_topology(model: "EcModel") -> None:
    """Remove prot_standard, usage_prot_standard, and the 'standard'
    gene from the cobra model (if present)."""
    cobra_rxn_ids = {r.id for r in model.reactions}
    if _STANDARD_USAGE_RXN_ID in cobra_rxn_ids:
        rxn = model.reactions.get_by_id(_STANDARD_USAGE_RXN_ID)
        model.remove_reactions([rxn])

    cobra_met_ids = {m.id for m in model.metabolites}
    if _STANDARD_MET_ID in cobra_met_ids:
        met = model.metabolites.get_by_id(_STANDARD_MET_ID)
        model.remove_metabolites([met])

    cobra_gene_ids = {g.id for g in model.genes}
    if _STANDARD_NAME in cobra_gene_ids:
        gene = model.genes.get_by_id(_STANDARD_NAME)
        # cobra removes the gene from the genes DictList directly.
        # Genes are typically tied to reactions via gene_reaction_rule;
        # since we just removed usage_prot_standard, the gene should be
        # orphaned and safe to remove.
        model.genes.remove(gene)
