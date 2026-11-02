"""FBA simulation half of the evotune kcat tuning distance function.

This module only does the FBA half: given an already kcat-constrained
``EcModel`` and one :class:`~geckopy.databases.flux_data.FluxData`
dataset (either ``evotune_data.flux_data`` or ``evotune_data.max_grate``),
simulate every condition and report growth + exchange fluxes.
:mod:`.distance` turns those raw numbers into an RMSE against the
matching experimental values -- kept separate so the RMSE math is
testable with hand-built inputs and no ``EcModel``/FBA at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from ...solver_support import optimize_model

if TYPE_CHECKING:
    from ...databases.flux_data import FluxData
    from ...ec_model.ec_model import EcModel


@dataclass
class ConditionSimResult:
    """One condition's simulated outcome.

    ``feasible=False`` (LP infeasible/unbounded, or no biomass flux)
    means ``growth``/``exch_fluxes``/``block_fluxes`` are meaningless;
    :mod:`.distance` applies the fixed infeasibility penalty for that
    row instead of reading them.
    """

    feasible: bool
    growth: float = float("nan")
    exch_fluxes: dict[str, float] = field(default_factory=dict)
    block_fluxes: dict[str, float] = field(default_factory=dict)


def simulate_evotune_dataset(
    model: "EcModel",
    flux_data: "FluxData",
    *,
    constrain: bool,
    zero_flux_rxns: list[str],
    bio_rxn_id: str,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
) -> list[ConditionSimResult]:
    """Simulate every condition in one tuning dataset.

    The steps are:

    1. Every exchange reaction that appears as a *condition name*
       anywhere in ``flux_data.conds`` (matched by name against
       ``flux_data.exch_mets``) is blocked (``lower_bound = 0``) up
       front, across the board -- not just the reaction for the row
       being solved -- so one row's carbon source can't leak flux into
       another row's simulation.
    2. For row ``i``, that row's own carbon-source reaction is
       unblocked: fixed at the measured uptake rate if ``constrain``
       is True (``evotune_data.flux_data``), or fully opened to ``-1000``
       if ``constrain`` is False (``evotune_data.max_grate`` -- "what's
       the *best possible* growth on this carbon source", not "growth
       at this measured rate").
    3. Optional per-condition adjustments run if the corresponding
       callable is supplied: anaerobic switch when the oxygen exchange
       column reads exactly 0, protein-content rescaling when
       ``Ptot`` is given and nonzero.
    4. Solve; record growth, plus (``constrain`` datasets only, since
       ``max_grate`` has no exchange measurements to compare against)
       every non-NaN measured exchange's simulated flux and every
       ``zero_flux_rxns`` entry's simulated flux.

    All mutation happens inside ``with model:``, which cobra reverts
    on exit -- only bounds are touched here, so this is safe to call
    repeatedly against one persistent (per-worker) model instance; a
    per-particle ``EcModel.copy()`` is not used (prohibitively
    expensive on a real-scale model).

    Parameters
    ----------
    model
        EcModel with the current particle's kcats already applied
        (via ``apply_kcat_constraints``).
    flux_data
        The dataset to simulate.
    constrain
        True for flux data (fix uptake at the measured rate), False
        for max-growth data (open uptake fully).
    zero_flux_rxns
        Reaction IDs assumed zero-flux in every condition
        (``evotune_data.zero_flux``); the row's own carbon source is
        skipped even if it's also listed here.
    bio_rxn_id
        The biomass reaction ID (``adapter.params.bio_rxn``).
    make_anaerobic, change_protein_biomass
        Optional adapter-specific hooks (MATLAB:
        ``modelAdapter.makeModelAnaerobic`` /
        ``.changeProteinBiomass``). geckopy has no generic,
        organism-agnostic port of these yet -- in MATLAB they're
        written per model-adapter subclass (e.g.
        ``YeastGEMAdapter.m``). Left as an injection point rather
        than a hard requirement: data with no anaerobic/``Ptot`` rows
        never needs them, and passing ``None`` for either just skips
        that adjustment instead of raising.

    Returns
    -------
    list[ConditionSimResult]
        One entry per row of ``flux_data.conds``, in order.

    Raises
    ------
    ValueError
        If any ``flux_data.conds`` entry can't be matched by name to
        an exchange metabolite in ``flux_data.exch_mets``.
    """
    cond_to_exch_idx = _match_conditions_to_exch(flux_data)
    all_condition_rxn_ids = sorted(
        {flux_data.exch_rxn_ids[j] for j in set(cond_to_exch_idx)}
    )
    o2_col = _find_col(flux_data.exch_mets, "oxygen")
    measured_cols = [
        j for j in range(len(flux_data.exch_rxn_ids))
    ] if constrain else []

    results: list[ConditionSimResult] = []
    for i, exch_idx in enumerate(cond_to_exch_idx):
        carbon_rxn_id = flux_data.exch_rxn_ids[exch_idx]
        with model:
            for rxn_id in all_condition_rxn_ids:
                model.reactions.get_by_id(rxn_id).lower_bound = 0.0

            carbon_rxn = model.reactions.get_by_id(carbon_rxn_id)
            if constrain:
                carbon_rxn.lower_bound = float(flux_data.exch_fluxes[i, exch_idx])
            else:
                carbon_rxn.lower_bound = -1000.0

            if o2_col is not None and flux_data.exch_fluxes[i, o2_col] == 0.0:
                if make_anaerobic is not None:
                    make_anaerobic(model)

            ptot = float(flux_data.p_tot[i])
            if not (np.isnan(ptot) or ptot == 0.0):
                if change_protein_biomass is not None:
                    change_protein_biomass(model, ptot)

            sol = optimize_model(model)
            if (
                sol.status != "optimal"
                or sol.objective_value is None
                or np.isnan(sol.objective_value)
            ):
                results.append(ConditionSimResult(feasible=False))
                continue

            growth = float(sol.fluxes.get(bio_rxn_id, 0.0))
            exch_fluxes: dict[str, float] = {}
            block_fluxes: dict[str, float] = {}
            if constrain:
                for j in measured_cols:
                    if np.isnan(flux_data.exch_fluxes[i, j]):
                        continue
                    rxn_id = flux_data.exch_rxn_ids[j]
                    exch_fluxes[rxn_id] = float(sol.fluxes.get(rxn_id, 0.0))
                for rxn_id in zero_flux_rxns:
                    if rxn_id == carbon_rxn_id:
                        continue
                    block_fluxes[rxn_id] = float(sol.fluxes.get(rxn_id, 0.0))
            results.append(
                ConditionSimResult(
                    feasible=True,
                    growth=growth,
                    exch_fluxes=exch_fluxes,
                    block_fluxes=block_fluxes,
                )
            )
    return results


def _match_conditions_to_exch(flux_data: "FluxData") -> list[int]:
    """Column index into ``exch_mets``/``exch_rxn_ids`` for each
    condition, matched by name. Mirrors MATLAB's
    ``ismember(data.conds, data.exchMets)`` plus its "cannot be
    matched" error."""
    met_to_idx: dict[str, int] = {}
    for j, met in enumerate(flux_data.exch_mets):
        met_to_idx.setdefault(met, j)
    missing = sorted({c for c in flux_data.conds if c not in met_to_idx})
    if missing:
        raise ValueError(
            'Carbon source(s) "'
            + "; ".join(missing)
            + '" in the provided fluxData or maxGrowth cannot be matched '
            "by name with an exchange reaction."
        )
    return [met_to_idx[c] for c in flux_data.conds]


def _find_col(exch_mets: list[str], name: str) -> Optional[int]:
    try:
        return exch_mets.index(name)
    except ValueError:
        return None
