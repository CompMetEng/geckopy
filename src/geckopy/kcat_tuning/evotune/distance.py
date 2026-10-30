"""Carbon- and condition-weighted RMSE distance for evotune kcat tuning.

:mod:`.simulate` does the FBA half; this module is pure numpy (plus one
small model-reading helper for the carbon weights) so the RMSE math is
directly testable against hand-computed expected values, no
``EcModel``/FBA involved.

The biomass reaction is passed explicitly as ``bio_rxn_id``
(``adapter.params.bio_rxn``) rather than matched by a literal name, and
:func:`dataset_rmse` builds its reaction-id-to-column lookup once.

Every zero-carbon exchange metabolite (O2, water, ...) is clamped up to
a carbon weight of 1, not 0, so those reactions' flux mismatches still
count toward the RMSE at unit weight rather than being zeroed out of it.
See :func:`compute_excarbon`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from .simulate import ConditionSimResult

if TYPE_CHECKING:
    from ...ec_model.ec_model import EcModel
    from .data import EvotuneData
    from ...databases.flux_data import FluxData

#: MATLAB's hardcoded assumption of the biomass reaction's carbon
#: content, in Cmmol/gDCW, used to put growth-rate deviations on the
#: same carbon-weighted scale as exchange-flux deviations.
BIOMASS_CARBON_EQUIV = 41.0

#: Fixed penalty substituted for an infeasible/failed condition's RMSE
#: ("clearly worse than any real RMSE" -- MATLAB never documents why
#: 99 specifically, just that it must dominate the mean).
INFEASIBLE_PENALTY = 99.0


def compute_excarbon(
    model: "EcModel", rxn_ids: Iterable[str], bio_rxn_id: str,
) -> dict[str, float]:
    """Per-reaction carbon-weight lookup for RMSE weighting.

    The biomass reaction gets :data:`BIOMASS_CARBON_EQUIV`. Every
    other reaction gets the carbon count of its single exchanged
    metabolite (via cobrapy's parsed ``Metabolite.elements``, not a
    hand-rolled formula parser). A missing/unparseable formula, or a
    computed carbon count of exactly 0 (e.g. O2, water), both fall
    back to 1 -- matching MATLAB's ``Ematrix(isnan(Ematrix)) = 1`` and
    ``excarbon(excarbon == 0) = 1`` respectively.

    Parameters
    ----------
    model
        Model providing the metabolite formulas.
    rxn_ids
        Reaction IDs to compute weights for (typically every
        reaction referenced by a tuning dataset: its exchange
        columns, its zero-flux list, and the biomass reaction).
    bio_rxn_id
        The biomass reaction ID.

    Returns
    -------
    dict[str, float]
    """
    excarbon: dict[str, float] = {}
    for rxn_id in rxn_ids:
        if rxn_id == bio_rxn_id:
            excarbon[rxn_id] = BIOMASS_CARBON_EQUIV
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        mets = list(rxn.metabolites)
        carbon = mets[0].elements.get("C") if len(mets) == 1 else None
        value = float(carbon) if carbon is not None else 1.0
        excarbon[rxn_id] = value if value != 0.0 else 1.0
    return excarbon


def dataset_rmse(
    flux_data: "FluxData",
    sims: list[ConditionSimResult],
    *,
    constrain: bool,
    excarbon: dict[str, float],
    bio_rxn_id: str,
    penalty: float = INFEASIBLE_PENALTY,
) -> tuple[float, np.ndarray]:
    """RMSE (mean, per-condition array) for one tuning dataset.

    Parameters
    ----------
    flux_data
        The dataset being scored (``evotune_data.flux_data`` or
        ``evotune_data.max_grate``).
    sims
        One :class:`~.simulate.ConditionSimResult` per
        ``flux_data.conds`` row, e.g. from
        :func:`~.simulate.simulate_evotune_dataset`.
    constrain
        Must match the value passed to
        :func:`~.simulate.simulate_evotune_dataset` for these
        ``sims``: True includes carbon-weighted exchange-flux terms
        (flux data), False scores growth alone (max-growth data).
    excarbon
        Carbon-weight lookup from :func:`compute_excarbon`, covering
        every reaction referenced by ``sims``' ``exch_fluxes`` /
        ``block_fluxes`` plus ``bio_rxn_id``. A reaction missing from
        the lookup falls back to a weight of 1.
    bio_rxn_id
        The biomass reaction ID (for reading ``flux_data.gr_rate``'s
        carbon-equivalent counterpart).
    penalty
        RMSE substituted for an infeasible condition.

    Returns
    -------
    rmse : float
        Weighted (by ``flux_data.evotune_rmse_weight``, if present)
        mean of the per-condition RMSEs.
    rmse_list : numpy.ndarray
        The per-condition RMSEs (post-penalty, pre-weighting), shape
        ``(len(flux_data.conds),)``.
    """
    n = len(flux_data.conds)
    if len(sims) != n:
        raise ValueError(
            f"sims has {len(sims)} entries; expected {n} to match "
            f"flux_data.conds."
        )
    rxn_to_col = {r: j for j, r in enumerate(flux_data.exch_rxn_ids)}

    rmse_list = np.empty(n, dtype=float)
    for i, sim in enumerate(sims):
        if not sim.feasible:
            rmse_list[i] = penalty
            continue

        bio_meas = float(flux_data.gr_rate[i]) * BIOMASS_CARBON_EQUIV
        bio_sim = sim.growth * BIOMASS_CARBON_EQUIV

        if not constrain:
            rmse_list[i] = abs(bio_meas - bio_sim)
            continue

        measured = [bio_meas]
        simulated = [bio_sim]
        for rxn_id, sim_flux in sim.exch_fluxes.items():
            w = excarbon.get(rxn_id, 1.0)
            measured.append(w * float(flux_data.exch_fluxes[i, rxn_to_col[rxn_id]]))
            simulated.append(w * sim_flux)
        for rxn_id, sim_flux in sim.block_fluxes.items():
            w = excarbon.get(rxn_id, 1.0)
            weighted_sim = w * sim_flux
            if weighted_sim == 0.0:
                # Already-correct zero predictions don't dilute the RMSE
                # (MATLAB: `blockSim(blockSim == 0) = []`).
                continue
            measured.append(0.0)
            simulated.append(weighted_sim)

        diff = np.asarray(measured, dtype=float) - np.asarray(simulated, dtype=float)
        rmse_list[i] = float(np.sqrt(np.mean(diff ** 2)))

    weighted = rmse_list
    if flux_data.evotune_rmse_weight is not None:
        weighted = rmse_list * flux_data.evotune_rmse_weight
    return float(np.mean(weighted)), rmse_list


def evotune_distance(
    evotune_data: "EvotuneData",
    *,
    flux_sims: list[ConditionSimResult] | None,
    max_grate_sims: list[ConditionSimResult] | None,
    excarbon: dict[str, float],
    bio_rxn_id: str,
    penalty: float = INFEASIBLE_PENALTY,
    max_growth_weight: float = 1.0,
) -> tuple[float, dict[str, np.ndarray]]:
    """Combine both datasets' RMSE into one score.

    ``flux_sims`` must be given (non-``None``) iff ``evotune_data.flux_data``
    is not ``None``, and likewise for ``max_grate_sims`` /
    ``evotune_data.max_grate``. A dataset absent from ``evotune_data`` doesn't
    contribute to the combined RMSE (MATLAB: a missing dataset's
    ``rmse_*`` stays ``[]`` and drops out of ``validIdx``).

    Parameters
    ----------
    max_growth_weight
        Relative weight of the max-growth dataset against the flux
        dataset, giving ``(rmse_flux + w * rmse_max_grate) / (w + 1)``.
        At 2 the eight max-growth conditions carry twice the weight of
        the 33 flux conditions, which is the intended emphasis here:
        the max-growth set is where the carbon-source diversity lives,
        while 30 of the 33 flux conditions are glucose.


    Returns
    -------
    rmse : float
        Weighted mean of whichever dataset RMSEs are present. ``NaN``
        if neither dataset is present.
    detail : dict[str, numpy.ndarray]
        Per-condition RMSE arrays, keyed ``"flux_data"`` /
        ``"max_grate"`` for whichever datasets were scored.
    """
    parts: list[float] = []
    weights: list[float] = []
    detail: dict[str, np.ndarray] = {}

    if evotune_data.flux_data is not None:
        if flux_sims is None:
            raise ValueError("evotune_data.flux_data is set but flux_sims is None.")
        rmse, per_cond = dataset_rmse(
            evotune_data.flux_data, flux_sims,
            constrain=True, excarbon=excarbon, bio_rxn_id=bio_rxn_id,
            penalty=penalty,
        )
        parts.append(rmse)
        weights.append(1.0)
        detail["flux_data"] = per_cond

    if evotune_data.max_grate is not None:
        if max_grate_sims is None:
            raise ValueError("evotune_data.max_grate is set but max_grate_sims is None.")
        rmse, per_cond = dataset_rmse(
            evotune_data.max_grate, max_grate_sims,
            constrain=False, excarbon=excarbon, bio_rxn_id=bio_rxn_id,
            penalty=penalty,
        )
        parts.append(rmse)
        weights.append(float(max_growth_weight))
        detail["max_grate"] = per_cond

    if parts:
        rmse = float(np.dot(parts, weights) / np.sum(weights))
    else:
        rmse = float("nan")
    return rmse, detail
