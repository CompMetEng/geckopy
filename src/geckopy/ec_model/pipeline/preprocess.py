"""Structural preprocessing of the GEM before enzyme extension.

Corresponds to stages 1 to 4 of makeEcModel in GECKO MATLAB. These
stages do not touch the `ec` substructure; they only reshape the
underlying model so subsequent stages operate on a clean irreversible form.

Ported from GECKO MATLAB: src/change_model/makeEcModel.m.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import cobra

# Stage 4 (convertToIrrev): delegated to raven-toolbox. The function originated
# in geckopy and was adopted upstream once raven-toolbox was created. Re-export
# under the same name so the rest of geckopy is unaffected.
from raven_toolbox.manipulation.irreversible import convert_to_irreversible

if TYPE_CHECKING:
    from ...adapter import ModelAdapter

__all__ = [
    "convert_to_irreversible",
    "remove_pseudoreaction_gprs",
    "invert_backwards_only_reactions",
]


def remove_pseudoreaction_gprs(
    model: "cobra.Model", adapter: "ModelAdapter"
) -> list[str]:
    """Clear gene-protein-reaction rules from pseudoreactions.

    A reaction is treated as a pseudoreaction if either:

    - its name contains the substring ``pseudoreaction`` (case-sensitive); or
    - its ID is listed in the first column of ``data/pseudoRxns.tsv``
      inside the adapter's project folder, if that file exists.

    Parameters
    ----------
    model
        A cobra.Model that will be mutated in place: the gene reaction
        rule of each matched reaction is cleared (empty string). This
        automatically updates the reaction's `.genes` set via cobrapy's
        GPR handling.
    adapter
        Provides the project path used to locate the optional TSV.

    Returns
    -------
    list of str
        Sorted IDs of reactions whose GPR was cleared. Intended for
        logging and test assertions.
    """
    pseudo_ids: set[str] = set()

    # By reaction name (MATLAB: contains(model.rxnNames,'pseudoreaction'))
    for rxn in model.reactions:
        if rxn.name and "pseudoreaction" in rxn.name:
            pseudo_ids.add(rxn.id)

    # By optional TSV (first column = reaction ID)
    tsv_path = adapter.params.path / "data" / "pseudoRxns.tsv"
    if tsv_path.is_file():
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0]:
                    pseudo_ids.add(parts[0])

    # Clear GPRs for matched reactions that exist in the model.
    model_rxn_ids = {r.id for r in model.reactions}
    cleared: list[str] = []
    for rxn_id in pseudo_ids & model_rxn_ids:
        rxn = model.reactions.get_by_id(rxn_id)
        rxn.gene_reaction_rule = ""
        cleared.append(rxn_id)

    return sorted(cleared)

def invert_backwards_only_reactions(model: "cobra.Model") -> list[str]:
    """Flip reactions that are constrained to only carry negative flux.

    A reaction with ``lb < 0`` and ``ub == 0`` is physically identical
    to a forward-only reaction with inverted stoichiometry and bounds
    ``[0, -lb]``. Rewriting it in that canonical form keeps downstream
    irreversibility handling simple.

    Reactions with ``lb == 0`` (already forward) or with both bounds
    negative or both positive are not touched.

    Parameters
    ----------
    model
        A cobra.Model, mutated in place.

    Returns
    -------
    list of str
        Sorted IDs of reactions that were inverted.
    """
    inverted: list[str] = []
    for rxn in model.reactions:
        if rxn.lower_bound < 0 and rxn.upper_bound == 0:
            # Flip stoichiometry: every coefficient gets negated.
            flipped = {m: -coef for m, coef in rxn.metabolites.items()}
            # add_metabolites with combine=False overwrites the existing values.
            rxn.add_metabolites(flipped, combine=False)
            # Swap bounds. Assign upper_bound first to avoid the
            # lb > ub transient state that cobrapy rejects.
            new_ub = -rxn.lower_bound
            rxn.lower_bound = 0.0
            rxn.upper_bound = new_ub
            inverted.append(rxn.id)
    return sorted(inverted)
