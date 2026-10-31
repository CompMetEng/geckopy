"""Constrain enzyme usage reactions by measured protein concentrations.

Ported from GECKO MATLAB:
src/geckomat/limit_proteins/constrainEnzConcs.m.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


from ..ec_model.constants import (
    POOL_ID,
    USAGE_PREFIX,
)

logger = logging.getLogger(__name__)


def constrain_enz_concs(
    model: "EcModel",
    *,
    remove_constraints: bool = False,
    restrict_to: list[str] | None = None,
) -> None:
    """Constrain ``usage_prot_<enzyme>`` reactions by ``model.ec.concs``.

    For each enzyme in ``model.ec.enzymes`` whose entry in
    ``model.ec.concs`` is non-NaN, the corresponding
    ``usage_prot_<enzyme>`` reaction's upper bound is set to that
    concentration. Enzymes with NaN concentrations have their usage
    reaction reset to the default ``upper_bound = 1000``, drawing
    freely from ``prot_pool``. Only the reaction's upper bound is
    changed; its stoichiometry (as set by
    ``add_protein_usage_reactions``) is left untouched.

    Ported from GECKO MATLAB:
    src/geckomat/limit_proteins/constrainEnzConcs.m.

    Parameters
    ----------
    model
        EcModel with ``model.ec.concs`` populated (typically by
        ``fill_enz_concs``) and the protein pool / usage reaction
        machinery already in place. Mutated in place.
    remove_constraints
        When True, all usage reactions are reset to their default
        ``upper_bound = 1000``, regardless of ``ec.concs``.
        ``ec.concs`` itself is left unchanged.
    restrict_to
        If supplied, only the listed enzyme uniprot IDs are
        processed (others are left untouched). Useful for incremental
        updates from ``Enzyme.concentration`` setters. Unknown IDs
        in the list are silently ignored. Default ``None`` =
        process every enzyme in ``model.ec.enzymes``.

    Raises
    ------
    ValueError
        If ``model.ec.concs`` holds a negative concentration for a
        processed enzyme (unless ``remove_constraints`` is set), if the
        ``prot_pool`` metabolite is missing, or if any
        ``usage_prot_<enzyme>`` reaction is missing for an enzyme
        in ``model.ec.enzymes`` (e.g. because the model is
        gecko-light, which has no usage reactions).
    """
    if POOL_ID not in {m.id for m in model.metabolites}:
        raise ValueError(
            f"Cannot find {POOL_ID!r} pseudometabolite. The protein "
            f"pool machinery must be set up before constraining enzyme "
            f"concentrations."
        )

    cobra_rxn_ids = {r.id for r in model.reactions}
    for enz in model.ec.enzymes:
        rxn_id = f"{USAGE_PREFIX}{enz}"
        if rxn_id not in cobra_rxn_ids:
            raise ValueError(
                f"Usage reaction {rxn_id!r} not found. Usage reactions "
                f"are added by add_protein_usage_reactions; gecko-light "
                f"models do not have them."
            )

    if restrict_to is None:
        index_iter = enumerate(model.ec.enzymes)
    else:
        restrict_set = set(restrict_to)
        index_iter = (
            (i, enz)
            for i, enz in enumerate(model.ec.enzymes)
            if enz in restrict_set
        )

    index_list = list(index_iter)
    if not remove_constraints:
        negative = [
            enz for i, enz in index_list if model.ec.concs[i] < 0
        ]
        if negative:
            raise ValueError(
                "model.ec.concs has a negative concentration for "
                + ", ".join(negative) + "."
            )

    for i, enz in index_list:
        rxn = model.reactions.get_by_id(f"{USAGE_PREFIX}{enz}")
        rxn.upper_bound = 1000.0
        if remove_constraints:
            continue
        conc = model.ec.concs[i]
        if np.isnan(conc):
            continue
        rxn.upper_bound = float(conc)
