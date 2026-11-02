"""Tests for kcat_tuning.evotune.mask_size.

Reuses the same two-branch toy EcModel shape as test_evotune.py (kept
self-contained rather than imported, matching this test suite's
convention of one fixture builder per file).
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import cobra
from geckopy import EcModel, ModelAdapter
from geckopy.adapter.params import EvotuneParams
from geckopy.databases.flux_data import FluxData
from geckopy.ec_model.ec_data import EcData
from geckopy.kcat_tuning.evotune.data import EvotuneData
from geckopy.kcat_tuning.evotune.mask_size import (
    MaskSizePoint,
    recommend_target_impact_share,
    sweep_tunable_mask_size,
)
from geckopy.kcat_tuning.evotune.tuning import screen_kcat_leverage

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
    """Two independent branches, one kcat each -- same shape as
    test_evotune.py's fixture of the same name."""
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
        exch_fluxes=np.array([[-1000.0, np.nan], [np.nan, -1000.0]]),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
    )
    return EvotuneData(flux_data=None, max_grate=max_grate, zero_flux=[])


# --------------------------------------------------------------------------- #
# sweep_tunable_mask_size
# --------------------------------------------------------------------------- #


def test_sweep_reports_one_point_per_share_sorted(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=10, rmse_threshold=1e-6)

    points = sweep_tunable_mask_size(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        target_impact_shares=[1.0, 0.7, 0.9], seeds=[0], n_proc=1, verbose=False,
    )

    assert [p.target_impact_share for p in points] == [0.7, 0.9, 1.0]
    # This toy model's leverage is evenly split across exactly two
    # kcats (cum_leverage_share .667/1.0), so every share above two
    # thirds keeps both -- chosen to stay clear of
    # cmaes_kcat_tuning's own "too few free parameters" floor of 2.
    assert [p.n_selected for p in points] == [2, 2, 2]


def test_sweep_does_not_mutate_input_model(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=10, rmse_threshold=1e-6)
    before = model.ec.kcat.copy()

    sweep_tunable_mask_size(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        target_impact_shares=[1.0], seeds=[0], n_proc=1, verbose=False,
    )

    assert np.array_equal(model.ec.kcat, before)


def test_sweep_runs_every_seed_at_every_share(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=10, rmse_threshold=1e-6)

    points = sweep_tunable_mask_size(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        target_impact_shares=[1.0], seeds=[0, 1, 2], n_proc=1, verbose=False,
    )

    assert len(points) == 1
    assert points[0].seeds == (0, 1, 2)
    assert len(points[0].rmse) == 3
    # Different seeds explore a stochastic search differently; on this
    # toy problem they need not disagree by much, but nothing forces
    # every seed to find bit-identical optima.
    assert points[0].rmse_sd >= 0.0


def test_sweep_reuses_a_given_screen(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=10, rmse_threshold=1e-6)
    screen = screen_kcat_leverage(
        model, adapter=adapter, evotune_data=evotune_data, n_proc=1,
    )

    # select_tunable_mask reads cum_leverage_share directly; dropping it
    # breaks a screen that's actually used, but a call that silently
    # recomputes its own (correct) screen instead of reusing this
    # broken one would not notice.
    stripped = screen.drop(columns=["cum_leverage_share"])
    with pytest.raises(KeyError):
        sweep_tunable_mask_size(
            model, adapter=adapter, params=params, evotune_data=evotune_data,
            screen=stripped, target_impact_shares=[1.0], seeds=[0],
            n_proc=1, verbose=False,
        )


# --------------------------------------------------------------------------- #
# recommend_target_impact_share
# --------------------------------------------------------------------------- #


def _pt(share, rmse, seeds=None):
    rmse = tuple(rmse) if not isinstance(rmse, tuple) else rmse
    return MaskSizePoint(
        target_impact_share=share, n_selected=int(share * 100),
        rmse=rmse, seeds=seeds or tuple(range(len(rmse))),
    )


def test_recommend_picks_the_flattening_point():
    points = [
        _pt(0.5, (1.0,)),
        _pt(0.7, (0.5,)),
        _pt(0.8, (0.495,)),
        _pt(0.9, (0.49,)),
    ]
    share, reason = recommend_target_impact_share(points, rel_tol=0.05)
    assert share == 0.7
    assert "improve" in reason


def test_recommend_requires_at_least_two_points():
    with pytest.raises(ValueError):
        recommend_target_impact_share([_pt(0.9, (0.5,))])


def test_recommend_returns_largest_share_if_still_improving():
    points = [_pt(0.5, (1.0,)), _pt(0.7, (0.5,)), _pt(0.9, (0.1,))]
    share, reason = recommend_target_impact_share(points, rel_tol=0.01)
    assert share == 0.9
    assert "kept improving" in reason


def test_recommend_treats_a_later_regression_as_no_improvement():
    # A later share that's actually worse must not be picked over an
    # earlier, better one -- this is what happened in practice (the
    # motivating yeast N-sweep: N=1029 fit worse than N=500).
    points = [_pt(0.7, (1.0,)), _pt(0.8, (0.5,)), _pt(0.9, (0.6,))]
    share, reason = recommend_target_impact_share(points, rel_tol=0.05)
    assert share == 0.8


def test_recommend_noise_floor_stops_a_real_but_small_gain_from_mattering():
    # Two seeds at the earlier point give it a real (if crude) sd
    # estimate; a later point barely inside that noise should not look
    # like a genuine improvement even though rel_tol alone might allow it.
    points = [
        _pt(0.7, (0.70, 0.74)),  # mean 0.72, sd 0.028
        _pt(0.9, (0.71,)),        # improvement of 0.01, well inside 2*sd
    ]
    # rel_tol tighter than the ~1.4% relative gain, so only the
    # noise-floor check (not the relative-tolerance one) can be why
    # this doesn't count as an improvement.
    share, reason = recommend_target_impact_share(
        points, rel_tol=0.005, noise_multiplier=2.0,
    )
    assert share == 0.7
    assert "noise" in reason


def test_recommend_sorts_points_regardless_of_input_order():
    points = [_pt(0.9, (0.49,)), _pt(0.5, (1.0,)), _pt(0.7, (0.5,))]
    share, _ = recommend_target_impact_share(points, rel_tol=0.05)
    assert share == 0.7


# --------------------------------------------------------------------------- #
# MaskSizePoint
# --------------------------------------------------------------------------- #


def test_mask_size_point_sd_is_nan_with_one_seed():
    p = _pt(0.9, (0.5,))
    assert np.isnan(p.rmse_sd)


def test_mask_size_point_mean_and_sd_with_multiple_seeds():
    p = _pt(0.9, (0.4, 0.5, 0.6))
    assert p.rmse_mean == pytest.approx(0.5)
    assert p.rmse_sd == pytest.approx(np.std([0.4, 0.5, 0.6], ddof=1))


# --------------------------------------------------------------------------- #
# rmse_trace's last entry isn't always its lowest (only true when
# prior_penalty_weight=0 -- the default is 0.03, so this is the common
# case, not an edge case): sweep_tunable_mask_size must use the
# trace's minimum, not its last entry.
# --------------------------------------------------------------------------- #


def test_sweep_uses_rmse_trace_minimum_not_last_entry(tmp_path, monkeypatch):
    from geckopy.kcat_tuning.evotune import mask_size as mask_size_module
    from geckopy.kcat_tuning.evotune.tuning import EvotuneResult

    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    evotune_data = _evotune_data()
    params = EvotuneParams(max_generations=10, rmse_threshold=1e-6)

    # A later generation can improve the (prior-penalised) objective
    # while its own plain RMSE is worse than an earlier generation's --
    # cmaes_kcat_tuning's own code only records rmse_trace alongside an
    # objective improvement, so the trace need not be monotonic in RMSE
    # alone. 0.71934 here, not the last entry 0.72331, is the value
    # that was actually achieved.
    fake_trace = [1.0, 0.8, 0.71934, 0.72331]

    def fake_cmaes_kcat_tuning(run_model, **kwargs):
        return EvotuneResult(
            rxns=["R_glc", "R_eth"], old_kcat=run_model.ec.kcat.copy(),
            new_kcat=run_model.ec.kcat.copy(), groups=["brenda", "dlkcat"],
            rmse_trace=fake_trace, objective_trace=fake_trace,
            n_generations=len(fake_trace), converged=False,
        )

    monkeypatch.setattr(mask_size_module, "cmaes_kcat_tuning", fake_cmaes_kcat_tuning)

    points = sweep_tunable_mask_size(
        model, adapter=adapter, params=params, evotune_data=evotune_data,
        target_impact_shares=[1.0], seeds=[0], n_proc=1, verbose=False,
    )

    assert points[0].rmse == (0.71934,)
