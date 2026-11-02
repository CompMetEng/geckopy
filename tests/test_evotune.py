"""Tests for kcat_tuning.evotune.tuning.

Two toy EcModels:

- ``_build_toy`` -- two independent enzymes (no isozymes), one per
  carbon source, so each condition's growth depends on exactly one
  kcat. Used for the basic screen/select/tune correctness checks.
- ``_build_tied_toy`` -- two isozyme copies of one reaction, sharing a
  prior and a source, drawing from one shared enzyme budget. Used to
  check that ``tie_isozymes`` actually changes what gets tuned.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cobra
from geckopy import EcModel, ModelAdapter
from geckopy.adapter.params import EvotuneParams
from geckopy.databases.flux_data import FluxData
from geckopy.ec_model.ec_data import EcData
from geckopy.kcat_tuning.evotune.tuning import (
    cmaes_kcat_tuning,
    screen_kcat_leverage,
    select_tunable_mask,
    tune_prior_penalty_weight,
)
from geckopy.kcat_tuning.evotune.data import EvotuneData
from geckopy.kcat_tuning.evotune.tuning import EvotuneResult

_TRUE_KCAT = 2.0
_START_KCAT = 1.0
_MW = 100.0
_TRUE_GROWTH = _TRUE_KCAT * 3600.0 / _MW


def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'bio_rxn = "biomass"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_toy(adapter: ModelAdapter) -> EcModel:
    """Two independent branches, one kcat each."""
    model = EcModel("toy", adapter=adapter)

    glc_e = cobra.Metabolite("glc_e", compartment="e")
    eth_e = cobra.Metabolite("eth_e", compartment="e")
    prot_pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_glc = cobra.Metabolite("prot_Eglc", compartment="c")
    prot_eth = cobra.Metabolite("prot_Eeth", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    model.add_metabolites([glc_e, eth_e, prot_pool, prot_glc, prot_eth, bio_met])

    EX_glc = cobra.Reaction("EX_glc")
    EX_glc.add_metabolites({glc_e: -1.0})
    EX_glc.bounds = (-1000.0, 0.0)

    EX_eth = cobra.Reaction("EX_eth")
    EX_eth.add_metabolites({eth_e: -1.0})
    EX_eth.bounds = (-1000.0, 0.0)

    coeff = _MW / (_START_KCAT * 3600.0)
    R_glc = cobra.Reaction("R_glc")
    R_glc.add_metabolites({glc_e: -1.0, prot_glc: -coeff, bio_met: 1.0})
    R_glc.bounds = (0.0, 1000.0)

    R_eth = cobra.Reaction("R_eth")
    R_eth.add_metabolites({eth_e: -1.0, prot_eth: -coeff, bio_met: 1.0})
    R_eth.bounds = (0.0, 1000.0)

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.bounds = (0.0, 1000.0)

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({prot_pool: 1.0})
    pool_ex.bounds = (0.0, 1.0)

    usage_glc = cobra.Reaction("usage_prot_Eglc")
    usage_glc.add_metabolites({prot_pool: -1.0, prot_glc: 1.0})
    usage_glc.bounds = (0.0, 1000.0)

    usage_eth = cobra.Reaction("usage_prot_Eeth")
    usage_eth.add_metabolites({prot_pool: -1.0, prot_eth: 1.0})
    usage_eth.bounds = (0.0, 1000.0)

    model.add_reactions(
        [EX_glc, EX_eth, R_glc, R_eth, BIO, pool_ex, usage_glc, usage_eth]
    )
    model.objective = "biomass"

    model.ec = EcData(
        rxns=["R_glc", "R_eth"],
        kcat=np.array([_START_KCAT, _START_KCAT]),
        source=["brenda", "dlkcat"],
        notes=["", ""],
        eccodes=["", ""],
        genes=["g_glc", "g_eth"],
        enzymes=["Eglc", "Eeth"],
        mw=np.array([_MW, _MW]),
        sequence=["", ""],
        concs=np.array([np.nan, np.nan]),
        rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    return model


def _evotune_data() -> EvotuneData:
    max_grate = FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_TRUE_GROWTH, _TRUE_GROWTH]),
        exch_fluxes=np.array(
            [[-1000.0, np.nan], [np.nan, -1000.0]]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
    )
    return EvotuneData(flux_data=None, max_grate=max_grate, zero_flux=[])


def _build_tied_toy(adapter: ModelAdapter) -> EcModel:
    """Two isozyme copies of one reaction (sharing a prior, a source,
    and one enzyme budget -- either can carry all the flux), plus one
    independent branch. The independent branch exists only so that
    tying still leaves two free parameters: cma's own N=1 path errors
    (verified directly, not assumed), so a model that tied down to a
    single parameter couldn't exercise cmaes_kcat_tuning end to end."""
    model = EcModel("tied_toy", adapter=adapter)

    glc_e = cobra.Metabolite("glc_e", compartment="e")
    eth_e = cobra.Metabolite("eth_e", compartment="e")
    prot_pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_1 = cobra.Metabolite("prot_Eiso1", compartment="c")
    prot_2 = cobra.Metabolite("prot_Eiso2", compartment="c")
    prot_eth = cobra.Metabolite("prot_Eeth", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    model.add_metabolites(
        [glc_e, eth_e, prot_pool, prot_1, prot_2, prot_eth, bio_met]
    )

    EX_glc = cobra.Reaction("EX_glc")
    EX_glc.add_metabolites({glc_e: -1.0})
    EX_glc.bounds = (-1000.0, 0.0)

    EX_eth = cobra.Reaction("EX_eth")
    EX_eth.add_metabolites({eth_e: -1.0})
    EX_eth.bounds = (-1000.0, 0.0)

    coeff = _MW / (_START_KCAT * 3600.0)
    R_iso1 = cobra.Reaction("R_iso_EXP_1")
    R_iso1.add_metabolites({glc_e: -1.0, prot_1: -coeff, bio_met: 1.0})
    R_iso1.bounds = (0.0, 1000.0)

    R_iso2 = cobra.Reaction("R_iso_EXP_2")
    R_iso2.add_metabolites({glc_e: -1.0, prot_2: -coeff, bio_met: 1.0})
    R_iso2.bounds = (0.0, 1000.0)

    R_eth = cobra.Reaction("R_eth")
    R_eth.add_metabolites({eth_e: -1.0, prot_eth: -coeff, bio_met: 1.0})
    R_eth.bounds = (0.0, 1000.0)

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.bounds = (0.0, 1000.0)

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({prot_pool: 1.0})
    pool_ex.bounds = (0.0, 1.0)

    usage_1 = cobra.Reaction("usage_prot_iso1")
    usage_1.add_metabolites({prot_pool: -1.0, prot_1: 1.0})
    usage_1.bounds = (0.0, 1000.0)

    usage_2 = cobra.Reaction("usage_prot_iso2")
    usage_2.add_metabolites({prot_pool: -1.0, prot_2: 1.0})
    usage_2.bounds = (0.0, 1000.0)

    usage_eth = cobra.Reaction("usage_prot_Eeth")
    usage_eth.add_metabolites({prot_pool: -1.0, prot_eth: 1.0})
    usage_eth.bounds = (0.0, 1000.0)

    model.add_reactions([
        EX_glc, EX_eth, R_iso1, R_iso2, R_eth, BIO, pool_ex,
        usage_1, usage_2, usage_eth,
    ])
    model.objective = "biomass"

    model.ec = EcData(
        rxns=["R_iso_EXP_1", "R_iso_EXP_2", "R_eth"],
        kcat=np.array([_START_KCAT, _START_KCAT, _START_KCAT]),
        source=["brenda", "brenda", "dlkcat"],
        notes=["", "", ""],
        eccodes=["", "", ""],
        genes=["g_iso1", "g_iso2", "g_eth"],
        enzymes=["Eiso1", "Eiso2", "Eeth"],
        mw=np.array([_MW, _MW, _MW]),
        sequence=["", "", ""],
        concs=np.array([np.nan, np.nan, np.nan]),
        rxn_enz_mat=sparse.csr_matrix(np.eye(3)),
    )
    return model


def _tied_evotune_data() -> EvotuneData:
    max_grate = FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_TRUE_GROWTH, _TRUE_GROWTH]),
        exch_fluxes=np.array(
            [[-1000.0, np.nan], [np.nan, -1000.0]]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
    )
    return EvotuneData(flux_data=None, max_grate=max_grate, zero_flux=[])


def test_screen_ranks_both_load_bearing_kcats(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()

    screen = screen_kcat_leverage(
        model, adapter=adapter, evotune_data=evotune_data, n_proc=1,
    )

    assert set(screen["rxn_id"]) == {"R_glc", "R_eth"}
    assert (screen["leverage"] > 0).all()
    assert (screen["n_isozymes"] == 1).all()
    # Sorted by rank descending, cumulative share reaches 1.0.
    assert list(screen["rank"]) == sorted(screen["rank"], reverse=True)
    assert screen["cum_leverage_share"].iloc[-1] == pytest.approx(1.0)


def test_select_tunable_mask_is_relative_and_monotone(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    screen = screen_kcat_leverage(
        model, adapter=adapter, evotune_data=evotune_data, n_proc=1,
    )

    tiny = select_tunable_mask(model, screen, target_impact_share=0.01)
    full = select_tunable_mask(model, screen, target_impact_share=1.0)

    # A minimal target still keeps at least the top-ranked kcat.
    assert tiny.sum() == 1
    assert tiny[list(model.ec.rxns).index(screen["rxn_id"].iloc[0])]
    # A target of 1.0 keeps everything with nonzero leverage.
    assert full.sum() == 2
    # Raising the target never drops something a lower target kept.
    assert np.all(full[tiny])


def test_select_tunable_mask_empty_screen_returns_empty_mask(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    empty = select_tunable_mask(model, pd.DataFrame())
    assert empty.sum() == 0
    assert empty.shape == model.ec.kcat.shape


def test_cmaes_tuning_moves_kcats_toward_truth(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=15, rmse_threshold=1e-6)
    before = model.ec.kcat.copy()

    result = cmaes_kcat_tuning(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        n_proc=1, seed=0, verbose=False,
    )

    assert isinstance(result, EvotuneResult)
    assert result.rxns == ["R_glc", "R_eth"]
    assert result.groups == ["brenda", "dlkcat"]
    assert np.array_equal(result.old_kcat, before)
    assert np.array_equal(model.ec.kcat, result.new_kcat)
    assert len(result.rmse_trace) == result.n_generations > 0

    # Both kcats should end up much closer to the true value than they
    # started (both start at exactly half of it).
    for i in range(2):
        assert abs(result.new_kcat[i] - _TRUE_KCAT) < abs(before[i] - _TRUE_KCAT)


def test_cmaes_tuning_n_proc_matches_serial(tmp_path):
    params = EvotuneParams(max_generations=5, rmse_threshold=-1.0)

    def _run(n_proc):
        adapter = _adapter(tmp_path)
        model = _build_toy(adapter)
        evotune_data = _evotune_data()
        return cmaes_kcat_tuning(
            model, adapter=adapter, params=params, evotune_data=evotune_data,
            n_proc=n_proc, seed=0, popsize=6, verbose=False,
        )

    serial = _run(1)
    parallel = _run(2)

    assert np.array_equal(serial.new_kcat, parallel.new_kcat)
    # rmse_trace values are summed in a different order across worker
    # processes than in-process, so they can differ by a couple of ULPs
    # even when the optimization itself (new_kcat) is bit-identical.
    assert np.allclose(serial.rmse_trace, parallel.rmse_trace, rtol=1e-9, atol=1e-12)
    assert serial.n_generations == parallel.n_generations


def test_tie_isozymes_forces_equal_values(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_tied_toy(adapter)
    evotune_data = _tied_evotune_data()
    params = EvotuneParams(max_generations=8, rmse_threshold=-1.0,
                            tie_isozymes=True)

    screen = screen_kcat_leverage(
        model, adapter=adapter, params=params, evotune_data=evotune_data, n_proc=1,
    )
    # Two rows: the tied isozyme pair as one group, R_eth as another.
    assert len(screen) == 2
    assert sorted(screen["n_isozymes"]) == [1, 2]

    result = cmaes_kcat_tuning(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        n_proc=1, seed=0, verbose=False,
    )
    assert result.rxns == ["R_iso_EXP_1", "R_iso_EXP_2", "R_eth"]
    assert result.new_kcat[0] == result.new_kcat[1]


def test_tie_isozymes_false_keeps_groups_separate(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_tied_toy(adapter)
    evotune_data = _tied_evotune_data()
    params = EvotuneParams(tie_isozymes=False)

    screen = screen_kcat_leverage(
        model, adapter=adapter, params=params, evotune_data=evotune_data, n_proc=1,
    )
    assert len(screen) == 3
    assert (screen["n_isozymes"] == 1).all()


def test_no_tunable_kcats_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    model.ec.kcat[:] = 0.0
    evotune_data = _evotune_data()
    with pytest.raises(ValueError, match="No tunable kcats"):
        cmaes_kcat_tuning(model, adapter=adapter, evotune_data=evotune_data, n_proc=1)


def test_too_few_free_parameters_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    mask = np.array([True, False])
    with pytest.raises(ValueError, match="free parameter"):
        cmaes_kcat_tuning(
            model, adapter=adapter, evotune_data=evotune_data,
            tunable_mask=mask, n_proc=1,
        )


def test_tune_prior_penalty_weight_shape_and_fit_cost(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=6, rmse_threshold=-1.0)

    report = tune_prior_penalty_weight(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        candidates=(0.0, 1000.0), seeds=(0, 1), n_proc=1, verbose=False,
    )

    assert len(report) == 2
    assert list(report["prior_penalty_weight"]) == [0.0, 1000.0]
    assert (report["n_seeds"] == 2).all()
    # The best (lowest-distance) candidate has zero fit cost by construction.
    assert report["fit_cost"].min() == pytest.approx(0.0)
    assert (report["fit_cost"] >= 0.0).all()
    # A crushing penalty should move fewer kcats than none at all.
    unpenalised = report.loc[report["prior_penalty_weight"] == 0.0].iloc[0]
    crushed = report.loc[report["prior_penalty_weight"] == 1000.0].iloc[0]
    assert crushed["n_changed_mean"] <= unpenalised["n_changed_mean"]


def test_tune_prior_penalty_weight_restores_model(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    before = model.ec.kcat.copy()
    params = EvotuneParams(max_generations=4, rmse_threshold=-1.0)

    tune_prior_penalty_weight(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        candidates=(0.0,), seeds=(0,), n_proc=1, verbose=False,
    )

    # Unlike cmaes_kcat_tuning, this only reports -- it leaves the model
    # exactly as it found it.
    assert np.array_equal(model.ec.kcat, before)


def test_tune_prior_penalty_weight_single_seed_gives_nan_reproducibility(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=4, rmse_threshold=-1.0)

    report = tune_prior_penalty_weight(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        candidates=(0.0,), seeds=(0,), n_proc=1, verbose=False,
    )

    row = report.iloc[0]
    assert row["n_both_moved"] == 0
    assert np.isnan(row["pct_direction_agree"])
    assert np.isnan(row["median_fold_spread"])


def test_result_reports_its_own_corrections():
    """The result already carries everything the table needs."""
    res = EvotuneResult(
        rxns=["r_slow", "r_untouched", "r_drift"],
        old_kcat=np.array([1.0, 5.0, 2.0]),
        new_kcat=np.array([10.0, 5.0, 40.0]),
        groups=["brenda", "custom", "okp"],
    )
    rows = res.corrections(leverage=np.array([3.0, 1.0, 0.01]))

    # Ranked by leverage: the big drifter is last despite moving furthest.
    assert [r.rxn_id for r in rows] == ["r_slow", "r_drift"]
    assert rows[0].source == "brenda"
    assert rows[0].fold_change == pytest.approx(10.0)
    assert rows[0].cumulative_share == pytest.approx(3.0 / 4.01)


def test_screen_orders_rows_of_equal_rank_as_the_tie_groups(tmp_path, monkeypatch):
    from geckopy.kcat_tuning.evotune import tuning

    monkeypatch.setattr(tuning, "_score_kcat_vector", lambda *a, **k: (1.0, 1.0))
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    screen = screen_kcat_leverage(
        model, adapter=adapter, evotune_data=_evotune_data(), n_proc=1,
    )
    assert list(screen["leverage"]) == [0.0, 0.0]
    assert list(screen["rxn_id"]) == list(model.ec.rxns)


def test_cmaes_tuning_runs_all_generations_on_a_flat_objective(tmp_path, monkeypatch):
    from geckopy.kcat_tuning.evotune import tuning

    monkeypatch.setattr(tuning, "_score_kcat_vector", lambda *a, **k: (1.0, 1.0))
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    params = EvotuneParams(max_generations=6, rmse_threshold=-1.0)
    result = cmaes_kcat_tuning(
        model, adapter=adapter, params=params, evotune_data=_evotune_data(),
        n_proc=1, seed=0, verbose=False,
    )
    assert result.n_generations == 6


def test_screen_warns_when_the_model_has_no_formulas(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    for met in model.metabolites:
        met.formula = None
    with pytest.warns(UserWarning, match="no metabolite formulas"):
        screen_kcat_leverage(
            model, adapter=adapter, evotune_data=_evotune_data(), n_proc=1,
        )
