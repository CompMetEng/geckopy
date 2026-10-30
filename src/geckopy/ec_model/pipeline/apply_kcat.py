"""Apply kcat-derived stoichiometric constraints to an ecModel.

Ported from GECKO MATLAB: src/geckomat/change_model/applyKcatConstraints.m.
Supports both the full and gecko-light formulations. The two share the
clear-then-write idempotency contract but write to different metabolites:
full models write per-enzyme ``prot_<id>`` coefficients on each ec.rxn;
light models pick the lowest-cost isozyme per cobra reaction and write a
single ``prot_pool`` coefficient there.
"""
from __future__ import annotations

import contextlib
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..ec_model import EcModel


from ...solver_support import edit_on_fast_solver
from ..constants import POOL_ID, PROT_PREFIX
from .populate_ec import split_light_rxn_id


# Number of updated ec reactions from which a scipy model is edited on glpk.
_BULK_UPDATE = 100


def apply_kcat_constraints(
    model: "EcModel",
    update_rxns: list[str] | None = None,
) -> None:
    """Translate ec.kcat values into stoichiometric coefficients.

    For each reaction in ``ec.rxns`` with a valid kcat and at least one
    associated enzyme, writes a negative coefficient at
    ``S[prot_<enzyme>, rxn]`` of magnitude::

        subunits * MW / (kcat * 3600)

    where ``subunits`` is the corresponding entry of ``ec.rxn_enz_mat``
    (1 for a single subunit, higher for multi-copy complexes), ``MW``
    is the enzyme molecular weight from ``ec.mw`` (in Da), and the
    factor 3600 converts kcat from 1/s (its conventional unit and the
    unit used in ``ec.kcat``) to 1/h so it aligns with cobrapy's
    mmol/gDW/h flux convention.

    Entries with kcat == 0 or kcat == NaN are treated as "no kcat
    assigned"; their stoichiometric coefficients are set to 0 (i.e.,
    the enzyme places no constraint on that reaction).

    The function is idempotent: existing non-zero coefficients at
    ``S[prot_<enzyme>, rxn]`` are first cleared for the reactions being
    updated, then freshly written from current ec.kcat. Running it
    twice yields the same result as running it once. Setting a kcat
    back to NaN/0 and re-applying clears the previously written
    coefficient.

    Parameters
    ----------
    model
        An EcModel produced by ``make_ec_model``, with at least some
        kcat values assigned to ``model.ec.kcat``. Mutated in place.
    update_rxns
        Reaction IDs (from ``model.ec.rxns``) to update. ``None`` or an
        empty sequence updates all entries, as ``applyKcatConstraints``
        does with an empty ``updateRxns``; a caller that builds the list
        from a filter must skip the call when the filter finds nothing.

    Raises
    ------
    ValueError
        If ``update_rxns`` contains IDs not present in ``ec.rxns``.

    Warns
    -----
    UserWarning
        If every kcat to be applied is 0 or NaN. Old coefficients are
        still cleared in this case; nothing is written afterward.
    """
    # A scipy model is edited on glpk when many reactions change: the scipy interface
    # rewrites the rows of a constraint for every changed coefficient.
    many = update_rxns is None or len(update_rxns) == 0 or len(update_rxns) >= _BULK_UPDATE
    with edit_on_fast_solver(model) if many else contextlib.nullcontext():
        _apply_kcat_constraints(model, update_rxns)


def _apply_kcat_constraints(
    model: "EcModel",
    update_rxns: list[str] | None,
) -> None:
    if model.ec.gecko_light:
        _apply_kcat_constraints_light(model, update_rxns)
        return

    ec_rxn_ids = model.ec.rxns

    # Resolve update_rxns to a numpy array of ec-row indices.
    if update_rxns is not None:
        update_rxns = list(update_rxns)

    if not update_rxns:
        selected_idx = np.arange(len(ec_rxn_ids))
    else:
        id_to_idx = {rxn_id: i for i, rxn_id in enumerate(ec_rxn_ids)}
        unknown = [r for r in update_rxns if r not in id_to_idx]
        if unknown:
            raise ValueError(
                f"update_rxns contains IDs not present in ec.rxns: {unknown[:5]}"
            )
        selected_idx = np.array([id_to_idx[r] for r in update_rxns], dtype=int)

    if selected_idx.size == 0:
        return

    # Step 1: clear existing prot_<enzyme> coefficients on every reaction
    # in the update set. We never touch prot_pool here. This step always
    # runs regardless of kcat validity, so that flipping a kcat to NaN
    # and re-applying genuinely clears the old constraint.
    # The enzyme pseudometabolites are exactly prot_<accession> for the
    # accessions in ec.enzymes; deriving the set from there avoids scanning
    # every metabolite on each (often per-reaction) call. prot_pool is never
    # an ec.enzyme, so it is naturally excluded.
    prot_met_ids = {f"{PROT_PREFIX}{e}" for e in model.ec.enzymes}

    for idx in selected_idx:
        rxn_id = ec_rxn_ids[idx]
        rxn = model.reactions.get_by_id(rxn_id)
        to_clear = {
            m: 0.0
            for m in rxn.metabolites
            if m.id in prot_met_ids
        }
        if to_clear:
            rxn.add_metabolites(to_clear, combine=False)

    # Step 2: identify which selected entries have a real kcat (>0).
    kcats_selected = model.ec.kcat[selected_idx]
    valid = kcats_selected > 0

    if not valid.any():
        warnings.warn(
            "apply_kcat_constraints: ec.kcat has no real entries for the "
            "selected reactions; existing constraints were cleared and no "
            "new ones written.",
            UserWarning,
            stacklevel=2,
        )
        return

    # Step 3: write fresh coefficients for valid kcats.
    enz_idx_to_met_id = {
        i: f"{PROT_PREFIX}{enz}"
        for i, enz in enumerate(model.ec.enzymes)
    }
    rxn_enz_mat_csr = model.ec.rxn_enz_mat.tocsr()

    for local_i, idx in enumerate(selected_idx):
        if not valid[local_i]:
            continue

        kcat = float(kcats_selected[local_i])
        rxn_id = ec_rxn_ids[idx]
        rxn = model.reactions.get_by_id(rxn_id)

        row = rxn_enz_mat_csr.getrow(idx)
        enzyme_indices = row.indices
        subunit_counts = row.data

        if enzyme_indices.size == 0:
            continue

        updates: dict = {}
        for enz_idx, subunits in zip(enzyme_indices, subunit_counts):
            mw = float(model.ec.mw[enz_idx])
            coef = -subunits * mw / (kcat * 3600.0)
            met = model.metabolites.get_by_id(enz_idx_to_met_id[enz_idx])
            # Two enzyme rows can map to the same prot_<accession> metabolite
            # (distinct genes sharing one UniProt entry). Sum their demands
            # instead of letting the later one overwrite the earlier.
            updates[met] = updates.get(met, 0.0) + coef

        if updates:
            rxn.add_metabolites(updates, combine=False)


# --------------------------------------------------------------------------- #
# gecko-light branch
# --------------------------------------------------------------------------- #

def _apply_kcat_constraints_light(
    model: "EcModel",
    update_rxns: list[str] | None,
) -> None:
    """Write the lowest-cost-isozyme ``prot_pool`` coefficient on each
    affected cobra reaction.

    Light ec.rxns has one row per isozyme of each
    cobra reaction (distinguished by a 3-digit counter prefix). The light
    formulation collapses those rows to a single LP constraint per cobra
    reaction by picking the isozyme with the lowest ``MW_sum / kcat``
    cost, then writing ``-MW_sum / (kcat * 3600)`` as the ``prot_pool``
    coefficient. ``MW_sum`` is the sum of subunit MWs (the row's
    ``rxn_enz_mat`` slice dotted into ``ec.mw``).

    Cobra reactions whose chosen isozyme has ``kcat == 0`` (or whose
    every isozyme is kcat-less, or has no associated enzyme with a
    known MW) get any prior ``prot_pool`` coefficient cleared and no
    new one written.

    Parameters
    ----------
    model
        A gecko-light EcModel with ``ec.gecko_light is True``.
    update_rxns
        ec.rxns IDs (with the ``###_`` prefix) to update. ``None`` or an
        empty sequence means update every cobra reaction backed by at
        least one ec.rxns row. A reaction is updated as a whole: naming one of its
        isozyme rows recomputes the cheapest cost over all its rows.
    """
    ec_rxn_ids = model.ec.rxns

    if update_rxns is not None:
        update_rxns = list(update_rxns)

    if not update_rxns:
        selected_idx = np.arange(len(ec_rxn_ids))
    else:
        id_to_idx = {rxn_id: i for i, rxn_id in enumerate(ec_rxn_ids)}
        unknown = [r for r in update_rxns if r not in id_to_idx]
        if unknown:
            raise ValueError(
                f"update_rxns contains IDs not present in ec.rxns: "
                f"{unknown[:5]}"
            )
        selected_idx = np.array(
            [id_to_idx[r] for r in update_rxns], dtype=int,
        )

    if selected_idx.size == 0:
        return

    # The cobra reaction carries the LP constraint, so selecting one isozyme
    # row updates the reaction as a whole: every ec row of that reaction
    # competes for the cheapest cost, selected or not.
    affected = {
        split_light_rxn_id(ec_rxn_ids[idx])[1] for idx in selected_idx
    }
    rows_by_cobra: dict[str, list[int]] = defaultdict(list)
    for idx, ec_rxn_id in enumerate(ec_rxn_ids):
        _, cobra_id = split_light_rxn_id(ec_rxn_id)
        if cobra_id in affected:
            rows_by_cobra[cobra_id].append(idx)

    pool_met = model.metabolites.get_by_id(POOL_ID)
    rxn_enz_mat_csr = model.ec.rxn_enz_mat.tocsr()
    mw = model.ec.mw

    cleared_only = True
    for cobra_id, ec_rows in rows_by_cobra.items():
        try:
            rxn = model.reactions.get_by_id(cobra_id)
        except KeyError:
            continue

        # Step 1: clear any prior prot_pool coefficient on this reaction.
        # Done unconditionally so flipping every isozyme's kcat to 0 and
        # re-applying genuinely drops the old constraint.
        if pool_met in rxn.metabolites:
            rxn.add_metabolites({pool_met: 0.0}, combine=False)

        # Step 2: find the cheapest isozyme (smallest MW_sum / kcat).
        # Each ec row contributes (kcat, MW_sum). Rows with kcat <= 0 or
        # MW_sum == 0 (no enzyme matched in the coupling matrix) are
        # ignored. NaN MW values are treated as missing.
        best_cost = float("inf")
        best_mw_sum = 0.0
        best_kcat = 0.0
        for r in ec_rows:
            kcat = float(model.ec.kcat[r])
            if kcat <= 0:
                continue
            row = rxn_enz_mat_csr.getrow(r)
            if row.nnz == 0:
                continue
            mw_for_row = mw[row.indices]
            if np.any(np.isnan(mw_for_row)):
                continue
            mw_sum = float(np.sum(row.data * mw_for_row))
            if mw_sum <= 0:
                continue
            cost = mw_sum / kcat
            if cost < best_cost:
                best_cost = cost
                best_mw_sum = mw_sum
                best_kcat = kcat

        if best_kcat == 0.0:
            continue  # no valid isozyme; leave the reaction at cleared
        coef = -best_mw_sum / (best_kcat * 3600.0)
        rxn.add_metabolites({pool_met: coef}, combine=False)
        cleared_only = False

    if cleared_only:
        warnings.warn(
            "apply_kcat_constraints (gecko-light): ec.kcat has no real "
            "entries for the selected reactions; existing prot_pool "
            "constraints were cleared and no new ones written.",
            UserWarning,
            stacklevel=2,
        )
