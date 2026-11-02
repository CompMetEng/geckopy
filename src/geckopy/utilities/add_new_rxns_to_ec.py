"""Add new reactions and enzymes to an ecModel.

Ported from GECKO MATLAB:
src/geckomat/utilities/addNewRxnsToEC.m.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cobra
import numpy as np
from cobra.core.gene import GPR
from scipy import sparse

from ..ec_model.constants import (
    POOL_ID,
    PROT_PREFIX,
    USAGE_PREFIX,
)
from raven_toolbox.manipulation.expand import gpr_to_dnf
from ..ec_model.pipeline.protein_pool import _resolve_enzyme_compartment_id

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)


@dataclass
class NewEnzyme:
    """An enzyme to add alongside the new reactions.

    Attributes
    ----------
    enzyme
        UniProt ID. Used as the `<X>` in `prot_<X>` and
        `usage_prot_<X>`.
    gene
        Gene ID for `model.genes` and the usage rxn's GPR.
    mw
        Molecular weight (g/mmol) for `model.ec.mw`.
    """

    enzyme: str
    gene: str
    mw: float


@dataclass
class AddNewRxnsResult:
    """IDs of what was actually added (after isozyme/reversibility
    expansion and the dedupe of already-present enzymes).

    Attributes
    ----------
    rxns_added
        IDs of every cobra reaction actually added to the model,
        after isozyme (``_EXP_<n>``) and reversibility (``_REV``)
        expansion.
    enz_added
        UniProt IDs of the enzymes actually added (enzymes already
        present in `model.ec.enzymes` are excluded).
    """

    rxns_added: list[str] = field(default_factory=list)
    enz_added: list[str] = field(default_factory=list)


def add_new_rxns_to_ec(
    model: "EcModel",
    new_rxns: list[cobra.Reaction],
    new_enzymes: list[NewEnzyme],
) -> AddNewRxnsResult:
    """Add reactions and their enzymes to an ecModel.

    For each new reaction, the function:

    * Splits a reversible reaction (`lower_bound < 0`) into a forward
      copy plus a `_REV` copy with negated stoichiometry.
    * Splits an OR-of-ANDs GPR into one `_EXP_<n>` copy per AND-clause
      using the same DNF logic as `expand_model`.

    The combination of both splits gives all four variants when
    applicable (`R_EXP_1`, `R_EXP_2`, `R_REV_EXP_1`, `R_REV_EXP_2`).

    Each new enzyme contributes:

    * a `prot_<enzyme>` pseudometabolite in the enzyme compartment,
    * a `usage_prot_<enzyme>` reaction (`prot_pool -> prot_<enzyme>`,
      bounds `(0, 1000)`, GPR set to the gene),
    * an entry appended to `model.ec.enzymes`/`genes`/`mw`/
      `sequence`/`concs`,
    * a new column in `model.ec.rxn_enz_mat`.

    Each new reaction with a non-empty GPR also gets:

    * an entry appended to `model.ec.rxns` (with kcat=0, source=""),
    * a new row in `rxn_enz_mat` with 1.0 in the columns of every
      gene in its (single, post-split) AND-clause.

    Ported from GECKO MATLAB:
    src/geckomat/utilities/addNewRxnsToEC.m.

    Parameters
    ----------
    model
        Full EcModel (not gecko-light) with the protein pool /
        usage rxn machinery already installed. Mutated in place.
    new_rxns
        Reactions to add, as pre-built `cobra.Reaction` objects.
        Each must have stoichiometry, `gene_reaction_rule`,
        `lower_bound`, `upper_bound` set; build one from a string
        equation with cobra's `Reaction.build_reaction_from_string`
        if that's more convenient.
    new_enzymes
        Enzymes to add (one per UniProt ID). Enzymes already in
        `model.ec.enzymes` are silently skipped with a warning.

    Returns
    -------
    AddNewRxnsResult

    Raises
    ------
    NotImplementedError
        If `model.ec.gecko_light` is True.
    ValueError
        If a gene referenced in a new rxn's GPR is missing from
        both `model.genes` and `new_enzymes`.
    """
    if model.ec.gecko_light:
        raise NotImplementedError(
            "add_new_rxns_to_ec does not support gecko-light models."
        )
    from ..adapter import resolve_adapter
    adapter = resolve_adapter(
        model,
        purpose="add_new_rxns_to_ec reads params.enzyme_comp from the adapter",
    )

    new_enzymes = _filter_existing_enzymes(model, new_enzymes)
    enz_added = [e.enzyme for e in new_enzymes]

    _validate_genes_present(model, new_rxns, new_enzymes)

    expanded = []
    for rxn in new_rxns:
        expanded.extend(_expand_new_rxn(rxn))

    # Pre-check ids before any mutation, so a collision can't leave the model
    # half-modified (pseudometabolites added, reactions not).
    expanded_ids = [r.id for r in expanded]
    existing_rxn_ids = {r.id for r in model.reactions}
    collisions = sorted(set(expanded_ids) & existing_rxn_ids)
    if collisions:
        raise ValueError(
            f"{len(collisions)} new reaction id(s) already exist in the "
            f"model (examples: {collisions[:5]}). Use unique ids; expansion "
            f"adds _EXP_<n>/_REV suffixes."
        )
    dupes = sorted({rid for rid in expanded_ids if expanded_ids.count(rid) > 1})
    if dupes:
        raise ValueError(
            f"new reactions expand to duplicate ids (examples: {dupes[:5]})."
        )

    if new_enzymes:
        comp_id = _resolve_enzyme_compartment_id(
            model, adapter.params.enzyme_comp,
        )
        _add_protein_pseudometabolites(model, new_enzymes, comp_id)

    model.add_reactions(expanded)

    if new_enzymes:
        _add_usage_reactions(model, new_enzymes)
        _extend_ec_per_enzyme_fields(model, new_enzymes)

    _extend_ec_per_rxn_fields(model, expanded)

    return AddNewRxnsResult(
        rxns_added=[r.id for r in expanded],
        enz_added=enz_added,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _filter_existing_enzymes(
    model: "EcModel", new_enzymes: list[NewEnzyme],
) -> list[NewEnzyme]:
    existing = set(model.ec.enzymes)
    keep: list[NewEnzyme] = []
    skipped: list[str] = []
    for e in new_enzymes:
        if e.enzyme in existing:
            skipped.append(e.enzyme)
        else:
            keep.append(e)
    if skipped:
        logger.warning(
            "add_new_rxns_to_ec: enzymes %s already in model.ec.enzymes; "
            "skipping.", skipped,
        )
    return keep


def _validate_genes_present(
    model: "EcModel",
    new_rxns: list[cobra.Reaction],
    new_enzymes: list[NewEnzyme],
) -> None:
    """Every gene in any new_rxn's GPR must already be in
    `model.genes` or about to be added via `new_enzymes`."""
    available = {g.id for g in model.genes} | {e.gene for e in new_enzymes}
    missing: set[str] = set()
    for rxn in new_rxns:
        if not rxn.gene_reaction_rule:
            continue
        gpr = GPR.from_string(rxn.gene_reaction_rule)
        for gene_name in gpr.genes:
            if gene_name not in available:
                missing.add(gene_name)
    if missing:
        raise ValueError(
            f"Genes referenced in new_rxns' GPRs are missing from the "
            f"model and were not added via new_enzymes: "
            f"{sorted(missing)}"
        )


def _add_protein_pseudometabolites(
    model: "EcModel",
    new_enzymes: list[NewEnzyme],
    comp_id: str,
) -> None:
    new_mets = []
    for e in new_enzymes:
        met_id = f"{PROT_PREFIX}{e.enzyme}"
        if met_id in {m.id for m in model.metabolites}:
            continue
        m = cobra.Metabolite(met_id, name=met_id, compartment=comp_id)
        m.annotation["sbo"] = "SBO:0000252"
        m.notes["note"] = "Enzyme-usage pseudometabolite"
        new_mets.append(m)
    if new_mets:
        model.add_metabolites(new_mets)


def _expand_new_rxn(rxn: cobra.Reaction) -> list[cobra.Reaction]:
    """Apply isozyme + reversibility splitting and return the
    resulting cobra.Reaction objects (with the same stoichiometry,
    name, etc.)."""
    is_reversible = rxn.lower_bound < 0
    if rxn.gene_reaction_rule:
        clauses = gpr_to_dnf(rxn.gpr)
    else:
        clauses = []

    out: list[cobra.Reaction] = []
    if len(clauses) <= 1:
        # No isozyme split.
        out.append(_clone_forward(rxn, suffix=""))
        if is_reversible:
            out.append(_clone_reverse(rxn, suffix="_REV"))
    else:
        for i, clause in enumerate(clauses, start=1):
            forward = _clone_forward(
                rxn, suffix=f"_EXP_{i}", gpr_clause=clause,
            )
            out.append(forward)
            if is_reversible:
                rev = _clone_reverse(
                    rxn, suffix=f"_REV_EXP_{i}", gpr_clause=clause,
                )
                out.append(rev)
    return out


def _clone_forward(
    rxn: cobra.Reaction,
    *,
    suffix: str,
    gpr_clause: list[str] | None = None,
) -> cobra.Reaction:
    new = cobra.Reaction(
        id=f"{rxn.id}{suffix}",
        name=rxn.name,
    )
    new.lower_bound = max(0.0, rxn.lower_bound)
    new.upper_bound = rxn.upper_bound
    new.add_metabolites({m: c for m, c in rxn.metabolites.items()})
    if gpr_clause is not None:
        new.gene_reaction_rule = " and ".join(gpr_clause)
    elif rxn.gene_reaction_rule:
        new.gene_reaction_rule = rxn.gene_reaction_rule
    return new


def _clone_reverse(
    rxn: cobra.Reaction,
    *,
    suffix: str,
    gpr_clause: list[str] | None = None,
) -> cobra.Reaction:
    new = cobra.Reaction(
        id=f"{rxn.id}{suffix}",
        name=f"{rxn.name} (reversible)" if rxn.name else "",
    )
    new.lower_bound = 0.0
    new.upper_bound = -rxn.lower_bound  # original |lb|
    new.add_metabolites({m: -c for m, c in rxn.metabolites.items()})
    if gpr_clause is not None:
        new.gene_reaction_rule = " and ".join(gpr_clause)
    elif rxn.gene_reaction_rule:
        new.gene_reaction_rule = rxn.gene_reaction_rule
    return new


def _add_usage_reactions(
    model: "EcModel", new_enzymes: list[NewEnzyme],
) -> None:
    pool = model.metabolites.get_by_id(POOL_ID)
    new_rxns = []
    for e in new_enzymes:
        rxn_id = f"{USAGE_PREFIX}{e.enzyme}"
        if rxn_id in {r.id for r in model.reactions}:
            continue
        prot_met = model.metabolites.get_by_id(f"{PROT_PREFIX}{e.enzyme}")
        rxn = cobra.Reaction(rxn_id, name=rxn_id)
        rxn.lower_bound = 0.0
        rxn.upper_bound = 1000.0
        rxn.add_metabolites({pool: -1.0, prot_met: 1.0})
        rxn.gene_reaction_rule = e.gene
        new_rxns.append(rxn)
    if new_rxns:
        model.add_reactions(new_rxns)


def _extend_ec_per_enzyme_fields(
    model: "EcModel", new_enzymes: list[NewEnzyme],
) -> None:
    if not new_enzymes:
        return
    n_old_enz = model.ec.n_enzymes
    n_new_enz = len(new_enzymes)

    model.ec.enzymes = list(model.ec.enzymes) + [e.enzyme for e in new_enzymes]
    model.ec.genes = list(model.ec.genes) + [e.gene for e in new_enzymes]
    model.ec.mw = np.concatenate(
        [model.ec.mw, np.array([e.mw for e in new_enzymes], dtype=float)]
    )
    model.ec.sequence = list(model.ec.sequence) + [""] * n_new_enz
    if model.ec.concs.size == n_old_enz:
        model.ec.concs = np.concatenate(
            [model.ec.concs, np.full(n_new_enz, np.nan, dtype=float)]
        )

    n_rxns = model.ec.rxn_enz_mat.shape[0]
    new_cols = sparse.csr_matrix((n_rxns, n_new_enz), dtype=float)
    model.ec.rxn_enz_mat = sparse.hstack(
        [model.ec.rxn_enz_mat, new_cols], format="csr",
    )


def _extend_ec_per_rxn_fields(
    model: "EcModel", expanded_rxns: list[cobra.Reaction],
) -> None:
    rxns_with_gpr = [r for r in expanded_rxns if r.gene_reaction_rule]
    if not rxns_with_gpr:
        return

    n_new = len(rxns_with_gpr)
    n_enz = model.ec.n_enzymes

    enz_index = {e: i for i, e in enumerate(model.ec.genes)}

    new_row_block = sparse.lil_matrix((n_new, n_enz), dtype=float)
    for i, rxn in enumerate(rxns_with_gpr):
        # Each post-split rxn has a single AND-clause GPR.
        gene_names = [g.strip() for g in rxn.gene_reaction_rule.split(" and ")]
        for g in gene_names:
            j = enz_index.get(g)
            if j is not None:
                new_row_block[i, j] = 1.0

    model.ec.rxn_enz_mat = sparse.vstack(
        [model.ec.rxn_enz_mat, new_row_block.tocsr()], format="csr",
    )
    model.ec.rxns = list(model.ec.rxns) + [r.id for r in rxns_with_gpr]
    model.ec.kcat = np.concatenate(
        [model.ec.kcat, np.zeros(n_new, dtype=float)]
    )
    model.ec.source = list(model.ec.source) + [""] * n_new
    model.ec.notes = list(model.ec.notes) + [""] * n_new
    model.ec.eccodes = list(model.ec.eccodes) + [""] * n_new
