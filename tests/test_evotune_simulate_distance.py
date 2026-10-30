"""Tests for kcat_tuning.evotune.simulate / .distance.

Styled like test_sensitivity_tuning.py's hand-built toy EcModel: a
tiny two-carbon-source model with a known enzyme cap, so every
simulated growth/exchange-flux value used below is hand-predictable
from the model's stoichiometry rather than treated as a black box.
"""
from pathlib import Path

import cobra
import numpy as np
import pytest

from geckopy import EcModel, ModelAdapter
from geckopy.databases.flux_data import FluxData
from geckopy.kcat_tuning.evotune.data import EvotuneData
from geckopy.kcat_tuning.evotune.distance import (
    BIOMASS_CARBON_EQUIV,
    evotune_distance,
    compute_excarbon,
    dataset_rmse,
)
from geckopy.kcat_tuning.evotune.simulate import (
    ConditionSimResult,
    simulate_evotune_dataset,
)


# --------------------------------------------------------------------------- #
# Toy two-carbon-source ecModel fixture
# --------------------------------------------------------------------------- #
#
# glucose and ethanol each feed a shared pool metabolite A_c, which the
# single enzyme-gated reaction R turns into biomass at an enzyme-capped
# rate of kcat*3600/mw = 1*3600/100 = 36 (pool size 1 mg/gDW). R also
# co-produces a small fixed amount of a "byproduct" the experimental
# data assumes is never made (byp_c, 0.1 mol per mol of R flux) -- this
# exercises `zero_flux_rxns`' nonzero-simulated-flux case.
#
# - "glucose" condition: uptake unconstrained relative to the enzyme
#   cap (measured flux -36 exactly matches TR's capacity), so growth
#   is enzyme-limited: 36.
# - "ethanol" condition: uptake capped at -20 (below the enzyme cap),
#   so growth is carbon-limited instead: 20.

_KCAT = 1.0
_MW = 100.0
_ENZYME_CAP = _KCAT * 3600.0 / _MW  # 36.0
_BYPRODUCT_YIELD = 0.1


def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'bio_rxn = "biomass"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_toy(adapter: ModelAdapter) -> EcModel:
    model = EcModel("toy", adapter=adapter)

    glc_e = cobra.Metabolite("glc_e", compartment="e", formula="C6H12O6")
    eth_e = cobra.Metabolite("eth_e", compartment="e", formula="C2H6O")
    byp_c = cobra.Metabolite("byp_c", compartment="c", formula="C1")
    A_c = cobra.Metabolite("A_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    model.add_metabolites([glc_e, eth_e, byp_c, A_c, pool, prot_E, bio_met])

    EX_glc = cobra.Reaction("EX_glc")
    EX_glc.add_metabolites({glc_e: -1.0})
    EX_glc.bounds = (-1000.0, 0.0)

    EX_eth = cobra.Reaction("EX_eth")
    EX_eth.add_metabolites({eth_e: -1.0})
    EX_eth.bounds = (-1000.0, 0.0)

    EX_byp = cobra.Reaction("EX_byp")
    EX_byp.add_metabolites({byp_c: -1.0})
    EX_byp.bounds = (0.0, 1000.0)

    TR_glc = cobra.Reaction("TR_glc")
    TR_glc.add_metabolites({glc_e: -1.0, A_c: 1.0})
    TR_glc.bounds = (0.0, 1000.0)

    TR_eth = cobra.Reaction("TR_eth")
    TR_eth.add_metabolites({eth_e: -1.0, A_c: 1.0})
    TR_eth.bounds = (0.0, 1000.0)

    coeff = _MW / (_KCAT * 3600.0)  # mg/mmol per unit flux
    R = cobra.Reaction("R")
    R.add_metabolites(
        {A_c: -1.0, prot_E: -coeff, bio_met: 1.0, byp_c: _BYPRODUCT_YIELD}
    )
    R.bounds = (0.0, 1000.0)

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.bounds = (0.0, 1000.0)

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.bounds = (0.0, 1.0)  # 1 mg/gDW pool

    usage_E = cobra.Reaction("usage_prot_E")
    usage_E.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage_E.bounds = (0.0, 1000.0)

    model.add_reactions(
        [EX_glc, EX_eth, EX_byp, TR_glc, TR_eth, R, BIO, pool_ex, usage_E]
    )
    model.objective = "biomass"
    return model


def _flux_data(*, ethanol_grrate: float) -> FluxData:
    """Two conditions: glucose (enzyme-limited, everything matches
    exactly) and ethanol (carbon-limited; ``ethanol_grrate`` controls
    whether the measured growth matches the simulated 20.0 exactly or
    deliberately deviates from it)."""
    return FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_ENZYME_CAP, ethanol_grrate]),
        exch_fluxes=np.array(
            [
                [-_ENZYME_CAP, np.nan],
                [np.nan, -20.0],
            ]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
        evotune_rmse_weight=np.array([1.0, 1.0]),
        source=["test", "test"],
    )


def _max_grate_data() -> FluxData:
    """Max-growth dataset: uptake fully opened, so both conditions hit
    the same enzyme-capped growth of 36."""
    return FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_ENZYME_CAP, _ENZYME_CAP]),
        exch_fluxes=np.array(
            [
                [-1000.0, np.nan],
                [np.nan, -1000.0],
            ]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
    )


# --------------------------------------------------------------------------- #
# simulate_evotune_dataset
# --------------------------------------------------------------------------- #

def test_simulate_flux_data_enzyme_vs_carbon_limited(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    flux_data = _flux_data(ethanol_grrate=20.0)

    results = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )

    assert len(results) == 2
    glucose, ethanol = results

    # Glucose: enzyme-limited growth of 36, full carbon uptake used.
    assert glucose.feasible
    assert glucose.growth == pytest.approx(_ENZYME_CAP)
    assert glucose.exch_fluxes["EX_glc"] == pytest.approx(-_ENZYME_CAP)
    assert glucose.block_fluxes["EX_byp"] == pytest.approx(
        _ENZYME_CAP * _BYPRODUCT_YIELD
    )

    # Ethanol: carbon-limited to the -20 uptake cap, growth = 20.
    assert ethanol.feasible
    assert ethanol.growth == pytest.approx(20.0)
    assert ethanol.exch_fluxes["EX_eth"] == pytest.approx(-20.0)
    assert ethanol.block_fluxes["EX_byp"] == pytest.approx(20.0 * _BYPRODUCT_YIELD)

    # Bounds fully revert after the call (both `with model:` blocks exited).
    assert model.reactions.get_by_id("EX_glc").bounds == (-1000.0, 0.0)
    assert model.reactions.get_by_id("EX_eth").bounds == (-1000.0, 0.0)


def test_simulate_max_grate_opens_uptake_fully(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    max_grate = _max_grate_data()

    results = simulate_evotune_dataset(
        model, max_grate,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )

    assert [r.growth for r in results] == pytest.approx([_ENZYME_CAP, _ENZYME_CAP])
    # max_grate rows carry no exchange-flux measurements to report back.
    assert results[0].exch_fluxes == {}
    assert results[0].block_fluxes == {}


def test_simulate_unmatched_condition_name_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    flux_data = _flux_data(ethanol_grrate=20.0)
    flux_data.conds = ["glucose", "propionate"]

    with pytest.raises(ValueError, match="propionate"):
        simulate_evotune_dataset(
            model, flux_data,
            constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        )


# --------------------------------------------------------------------------- #
# compute_excarbon
# --------------------------------------------------------------------------- #

def test_compute_excarbon(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)

    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    assert excarbon["EX_glc"] == pytest.approx(6.0)   # C6H12O6
    assert excarbon["EX_eth"] == pytest.approx(2.0)   # C2H6O
    assert excarbon["EX_byp"] == pytest.approx(1.0)   # C1
    assert excarbon["biomass"] == pytest.approx(BIOMASS_CARBON_EQUIV)


def test_compute_excarbon_zero_carbon_clamps_to_one(tmp_path):
    """A genuinely zero-carbon exchanged metabolite (e.g. O2) still
    gets weight 1, not 0 -- MATLAB's `excarbon(excarbon==0)=1` quirk,
    ported deliberately so those reactions' mismatches aren't silently
    excluded from the RMSE."""
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    o2_e = cobra.Metabolite("o2_e", compartment="e", formula="O2")
    model.add_metabolites([o2_e])
    EX_o2 = cobra.Reaction("EX_o2")
    EX_o2.add_metabolites({o2_e: -1.0})
    EX_o2.bounds = (-1000.0, 0.0)
    model.add_reactions([EX_o2])

    excarbon = compute_excarbon(model, ["EX_o2"], bio_rxn_id="biomass")

    assert excarbon["EX_o2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# dataset_rmse / evotune_distance
# --------------------------------------------------------------------------- #

def test_dataset_rmse_measured_terms_match_only_byproduct_deviates(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    # ethanol_grrate=20.0 matches the ethanol condition's actual
    # carbon-limited growth exactly, and both conditions' own measured
    # exchange fluxes match the simulation exactly too -- the only
    # nonzero term left is the unmeasured byproduct (assumed
    # zero-flux, but nonzero in the simulation).
    flux_data = _flux_data(ethanol_grrate=20.0)
    sims = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, excarbon=excarbon, bio_rxn_id="biomass",
    )

    # The byproduct term is nonzero-simulated-vs-zero-measured, so RMSE
    # is not zero overall -- but every *measured* exchange/growth term
    # matches exactly, and the byproduct term is hand-computable.
    glucose_byp = _ENZYME_CAP * _BYPRODUCT_YIELD * 1.0  # carbon weight 1
    ethanol_byp = 20.0 * _BYPRODUCT_YIELD * 1.0
    # 3 terms per condition: growth (matches exactly -> 0), the
    # condition's own exchange flux (matches exactly -> 0), and the
    # byproduct block term (measured 0 vs the nonzero simulated value).
    expected_glucose_rmse = np.sqrt(np.mean([0.0, 0.0, glucose_byp ** 2]))
    expected_ethanol_rmse = np.sqrt(np.mean([0.0, 0.0, ethanol_byp ** 2]))

    assert per_cond[0] == pytest.approx(expected_glucose_rmse)
    assert per_cond[1] == pytest.approx(expected_ethanol_rmse)
    assert rmse == pytest.approx(np.mean([expected_glucose_rmse, expected_ethanol_rmse]))


def test_dataset_rmse_growth_mismatch_is_hand_computable(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    # Deliberately wrong measured growth for the ethanol condition
    # (18.0 vs the actual carbon-limited 20.0) -- a known, hand-provable
    # deviation on top of the byproduct term.
    flux_data = _flux_data(ethanol_grrate=18.0)
    sims = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    _, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, excarbon=excarbon, bio_rxn_id="biomass",
    )

    growth_diff = (18.0 - 20.0) * BIOMASS_CARBON_EQUIV  # bio_meas - bio_sim
    byp_diff = 0.0 - (20.0 * _BYPRODUCT_YIELD)  # measured 0 vs simulated
    expected = np.sqrt(np.mean([growth_diff ** 2, 0.0, byp_diff ** 2]))
    assert per_cond[1] == pytest.approx(expected)


def test_dataset_rmse_infeasible_condition_gets_penalty(tmp_path):
    flux_data = _flux_data(ethanol_grrate=20.0)
    sims = [ConditionSimResult(feasible=False), ConditionSimResult(feasible=False)]
    excarbon = {}

    rmse, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, excarbon=excarbon, bio_rxn_id="biomass", penalty=99.0,
    )

    assert list(per_cond) == [99.0, 99.0]
    assert rmse == pytest.approx(99.0)


def test_dataset_rmse_max_grate_scores_growth_only():
    max_grate = _max_grate_data()
    sims = [
        ConditionSimResult(feasible=True, growth=_ENZYME_CAP),
        ConditionSimResult(feasible=True, growth=30.0),  # deliberately off
    ]

    rmse, per_cond = dataset_rmse(
        max_grate, sims,
        constrain=False, excarbon={}, bio_rxn_id="biomass",
    )

    assert per_cond[0] == pytest.approx(0.0)
    expected_second = abs((_ENZYME_CAP - 30.0) * BIOMASS_CARBON_EQUIV)
    assert per_cond[1] == pytest.approx(expected_second)
    assert rmse == pytest.approx(np.mean([0.0, expected_second]))


def test_evotune_distance_combines_both_datasets(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    flux_data = _flux_data(ethanol_grrate=20.0)
    max_grate = _max_grate_data()
    evotune_data = EvotuneData(flux_data=flux_data, max_grate=max_grate, zero_flux=["EX_byp"])

    flux_sims = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    max_grate_sims = simulate_evotune_dataset(
        model, max_grate,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )
    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, detail = evotune_distance(
        evotune_data,
        flux_sims=flux_sims, max_grate_sims=max_grate_sims,
        excarbon=excarbon, bio_rxn_id="biomass",
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, excarbon=excarbon, bio_rxn_id="biomass",
    )
    expected_max_grate_rmse, _ = dataset_rmse(
        max_grate, max_grate_sims, constrain=False, excarbon=excarbon,
        bio_rxn_id="biomass",
    )
    assert rmse == pytest.approx(
        np.mean([expected_flux_rmse, expected_max_grate_rmse])
    )
    assert "flux_data" in detail and "max_grate" in detail


def test_evotune_distance_max_growth_weight_scales_the_max_growth_term(tmp_path):
    """``max_growth_weight`` scales the max-growth term, so 2 makes the
    eight max-growth conditions count double against the 33 flux ones."""
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    flux_data = _flux_data(ethanol_grrate=20.0)
    max_grate = _max_grate_data()
    evotune_data = EvotuneData(flux_data=flux_data, max_grate=max_grate, zero_flux=["EX_byp"])

    flux_sims = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    max_grate_sims = simulate_evotune_dataset(
        model, max_grate,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )
    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, excarbon=excarbon, bio_rxn_id="biomass",
    )
    expected_max_grate_rmse, _ = dataset_rmse(
        max_grate, max_grate_sims, constrain=False, excarbon=excarbon,
        bio_rxn_id="biomass",
    )

    rmse, _ = evotune_distance(
        evotune_data,
        flux_sims=flux_sims, max_grate_sims=max_grate_sims,
        excarbon=excarbon, bio_rxn_id="biomass", max_growth_weight=2.0,
    )
    assert rmse == pytest.approx(
        (expected_flux_rmse + 2.0 * expected_max_grate_rmse) / 3.0
    )

    # 0.5 reproduces MATLAB's maxGrowthWeight = 2, which weights flux.
    rmse_matlab, _ = evotune_distance(
        evotune_data,
        flux_sims=flux_sims, max_grate_sims=max_grate_sims,
        excarbon=excarbon, bio_rxn_id="biomass", max_growth_weight=0.5,
    )
    assert rmse_matlab == pytest.approx(
        (2.0 * expected_flux_rmse + expected_max_grate_rmse) / 3.0
    )

    # A lone dataset carries the whole score whatever the weight.
    solo = EvotuneData(flux_data=flux_data, max_grate=None, zero_flux=["EX_byp"])
    rmse_solo, _ = evotune_distance(
        solo,
        flux_sims=flux_sims, max_grate_sims=None,
        excarbon=excarbon, bio_rxn_id="biomass", max_growth_weight=2.0,
    )
    assert rmse_solo == pytest.approx(expected_flux_rmse)


def test_evotune_distance_missing_dataset_contributes_nothing(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    flux_data = _flux_data(ethanol_grrate=20.0)
    evotune_data = EvotuneData(flux_data=flux_data, max_grate=None, zero_flux=["EX_byp"])

    flux_sims = simulate_evotune_dataset(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    excarbon = compute_excarbon(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, detail = evotune_distance(
        evotune_data,
        flux_sims=flux_sims, max_grate_sims=None,
        excarbon=excarbon, bio_rxn_id="biomass",
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, excarbon=excarbon, bio_rxn_id="biomass",
    )
    assert rmse == pytest.approx(expected_flux_rmse)
    assert set(detail) == {"flux_data"}
