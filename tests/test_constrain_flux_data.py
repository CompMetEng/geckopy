"""Tests for apply_flux_data_constraints: the defaults taken from the adapter.

The bounds each setting gives are tested with the function in
raven_toolbox.experiments.
"""
from pathlib import Path

import cobra
import numpy as np
import pytest

from geckopy import EcModel, ModelAdapter
from geckopy.limit_proteins import apply_flux_data_constraints
from raven_toolbox.experiments import FluxData


def _adapter(tmp_path: Path, *, c_source: str = "EX_eth") -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'bio_rxn = "biomass"\n'
        f'c_source = "{c_source}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_model(adapter: ModelAdapter) -> EcModel:
    model = EcModel("toy", adapter=adapter)
    glc = cobra.Metabolite("glc_e", compartment="e")
    eth = cobra.Metabolite("eth_e", compartment="e")
    bio_met = cobra.Metabolite("bio", compartment="c")
    model.add_metabolites([glc, eth, bio_met])
    rxns = []
    for rxn_id, met, coeff in (
        ("EX_glc", glc, -1.0), ("EX_eth", eth, -1.0), ("biomass", bio_met, 1.0),
    ):
        rxn = cobra.Reaction(rxn_id)
        rxn.add_metabolites({met: coeff})
        rxn.bounds = (-1000.0, 1000.0)
        rxns.append(rxn)
    model.add_reactions(rxns)
    return model


def _flux_data() -> FluxData:
    return FluxData(
        conds=["c1"],
        p_tot=np.array([0.5]),
        gr_rate=np.array([0.4]),
        exch_fluxes=np.array([[-5.0]]),
        exch_mets=["EX_glc"],
        exch_rxn_ids=["EX_glc"],
    )


def test_biomass_and_carbon_source_come_from_the_adapter(tmp_path):
    model = _build_model(_adapter(tmp_path))

    apply_flux_data_constraints(model, _flux_data())

    assert model.reactions.get_by_id("biomass").upper_bound == pytest.approx(0.4)
    eth = model.reactions.get_by_id("EX_eth")
    assert (eth.lower_bound, eth.upper_bound) == (0.0, 0.0)


def test_empty_c_source_blocks_nothing(tmp_path):
    model = _build_model(_adapter(tmp_path))

    apply_flux_data_constraints(model, _flux_data(), c_source="")

    eth = model.reactions.get_by_id("EX_eth")
    assert (eth.lower_bound, eth.upper_bound) == (-1000.0, 1000.0)


def test_explicit_biomass_reaction_overrides_the_adapter(tmp_path):
    model = _build_model(_adapter(tmp_path))
    model.adapter = None

    apply_flux_data_constraints(model, _flux_data(), bio_rxn="biomass")

    assert model.reactions.get_by_id("biomass").upper_bound == pytest.approx(0.4)


def test_no_adapter_and_no_biomass_reaction_raises(tmp_path):
    model = _build_model(_adapter(tmp_path))
    model.adapter = None

    with pytest.raises(ValueError, match="adapter"):
        apply_flux_data_constraints(model, _flux_data())
