"""The experiment data of the full_ecModel tutorial can be scored.

Its first condition, ``Std``, has weight 0: the protein step reads it, and the
tuning leaves it out. ``Std`` matches no exchange metabolite, so scoring has to skip it
before the conditions are matched to exchange reactions.
"""
import warnings
from pathlib import Path

import cobra
import numpy as np
import pytest

from raven_toolbox.experiments import (
    compute_carbon_weights,
    dataset_rmse,
    load_experiment_data,
    simulate_conditions,
)

TUTORIAL_DATA = Path(__file__).parents[1] / "tutorials" / "full_ecModel" / "data"


def _model_with_exchanges(rxn_ids):
    """Every exchange reaction supplies its own metabolite, which a transport
    reaction turns into the intermediate that biomass consumes."""
    model = cobra.Model("tutorial_exchanges")
    inter = cobra.Metabolite("A_c", compartment="c")
    reactions = []
    for rxn_id in rxn_ids:
        met = cobra.Metabolite(f"m_{rxn_id}", compartment="e")
        exchange = cobra.Reaction(rxn_id)
        exchange.add_metabolites({met: -1.0})
        exchange.bounds = (-1000.0, 1000.0)
        transport = cobra.Reaction(f"T_{rxn_id}")
        transport.add_metabolites({met: -1.0, inter: 1.0})
        transport.bounds = (0.0, 1000.0)
        reactions += [exchange, transport]
    biomass = cobra.Reaction("biomass")
    biomass.add_metabolites({inter: -1.0})
    biomass.bounds = (0.0, 1000.0)
    model.add_reactions(reactions + [biomass])
    model.objective = "biomass"
    return model


@pytest.mark.skipif(not TUTORIAL_DATA.is_dir(), reason="tutorial data not in this checkout")
def test_the_tutorial_data_is_scored_without_its_weight_zero_condition():
    data = load_experiment_data(TUTORIAL_DATA)
    assert data.flux_data.conds[0] == "Std" and data.flux_data.weight[0] == 0
    rxn_ids = sorted(
        set(data.flux_data.exch_rxn_ids) | set(data.max_growth.exch_rxn_ids) | set(data.zero_flux)
    )
    model = _model_with_exchanges(rxn_ids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        carbon_weights = compute_carbon_weights(model, rxn_ids + ["biomass"], bio_rxn_id="biomass")

    flux_sims = simulate_conditions(
        model, data.flux_data, constrain=True, zero_flux_rxns=data.zero_flux,
        bio_rxn_id="biomass",
    )
    max_growth_sims = simulate_conditions(
        model, data.max_growth, constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )
    rmse, per_cond = dataset_rmse(
        data.flux_data, flux_sims, constrain=True, carbon_weights=carbon_weights,
    )
    max_growth_rmse, _ = dataset_rmse(
        data.max_growth, max_growth_sims, constrain=False, carbon_weights=carbon_weights,
    )

    assert flux_sims[0].skipped and not any(s.skipped for s in flux_sims[1:])
    assert np.isnan(per_cond[0]) and not np.isnan(per_cond[1:]).any()
    assert np.isfinite(rmse) and np.isfinite(max_growth_rmse)
