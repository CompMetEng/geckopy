"""Tests for kcat_tuning.evotune.settings_search and the min_leverage mask."""
import numpy as np
import pandas as pd
import pytest

from geckopy.adapter.params import EvotuneParams
from geckopy.kcat_tuning.evotune.settings_search import (
    SettingsRun,
    candidate_grid,
    recommend_tuning_settings,
    search_tuning_settings,
    summarise_runs,
)
from geckopy.kcat_tuning.evotune.tuning import screen_kcat_leverage, select_tunable_mask

from test_mask_size import GROUPS, _adapter, _build_toy, _experiment_data


def _run(settings, seed, rmse_trace, new, n_selected=4, old=None, gens=None, max_gens=100):
    old = np.ones(len(new)) if old is None else old
    return SettingsRun(
        settings=settings, seed=seed, n_selected=n_selected, old_kcat=np.asarray(old, float),
        new_kcat=np.asarray(new, float), rmse_trace=list(rmse_trace),
        n_generations=len(rmse_trace) if gens is None else gens, converged=False,
        max_generations=max_gens,
    )


# --- candidate_grid ---------------------------------------------------------


def test_candidate_grid_is_the_product():
    grid = candidate_grid(min_leverage=[0.02, 0.03], prior_penalty_weight=[0, 0.03])
    assert len(grid) == 4
    assert {"min_leverage": 0.03, "prior_penalty_weight": 0} in grid


# --- select_tunable_mask with min_leverage -------------------------------


def _screen():
    return pd.DataFrame({
        "rank": [0.09, 0.03, 0.006],
        "leverage": [0.3, 0.1, 0.06],
        "cum_leverage_share": [0.65, 0.87, 1.0],
        "_positions": [[0], [1], [2]],
    })


class _Ec:
    rxns = ["a", "b", "c"]


class _M:
    ec = _Ec()


def test_min_leverage_keeps_groups_at_or_above_the_weighted_cutoff():
    # rank / sigma_ref = 0.3, 0.1, 0.02
    mask = select_tunable_mask(_M(), _screen(), min_leverage=0.1, sigma_ref=0.3)
    assert mask.tolist() == [True, True, False]


def test_min_leverage_overrides_the_share():
    mask = select_tunable_mask(
        _M(), _screen(), target_impact_share=0.5, min_leverage=0.02, sigma_ref=0.3)
    assert mask.all()


def test_share_still_works_without_min_leverage():
    mask = select_tunable_mask(_M(), _screen(), target_impact_share=0.5)
    assert mask.tolist() == [True, False, False]


# --- summarise_runs ---------------------------------------------------------


def test_summary_one_row_per_candidate_with_agreement():
    a = {"min_leverage": 0.05}
    b = {"min_leverage": 0.01}
    runs = [
        _run(a, 0, [3, 2, 1.0], [4, 0.2, 1, 1], n_selected=2),
        _run(a, 1, [3, 2, 1.1], [3, 0.3, 1, 1], n_selected=2),
        _run(b, 0, [3, 2, 0.5], [4, 0.2, 5, 5], n_selected=4),
        _run(b, 1, [3, 2, 0.6], [4, 0.2, 0.2, 0.2], n_selected=4),
    ]
    table = summarise_runs(runs)
    assert list(table["n_selected"]) == [2, 4]
    small, large = table.iloc[0], table.iloc[1]
    assert small["rmse_mean"] == pytest.approx(1.05)
    assert small["pct_direction_agree"] == 1.0 and small["n_both_moved"] == 2
    assert large["pct_direction_agree"] == pytest.approx(0.5)
    assert small["n_seeds"] == 2 and not np.isnan(small["rmse_sd"])


def test_plateau_generation_and_cap():
    cand = {"min_leverage": 0.05}
    trace = [4.0, 2.0, 1.0] + [1.0] * 7
    table = summarise_runs([_run(cand, 0, trace, [2, 1], max_gens=10)])
    assert table.loc[0, "plateau_generation"] == 3
    assert bool(table.loc[0, "capped"])
    table = summarise_runs([_run(cand, 0, trace, [2, 1], max_gens=50)])
    assert not bool(table.loc[0, "capped"])


# --- recommend_tuning_settings ---------------------------------------------


def _table(rows):
    base = {"n_seeds": 2, "n_both_moved": 20, "pct_direction_agree": 0.95,
            "plateau_generation": 40, "capped": False, "seconds": 1.0, "rmse_sd": 0.01}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_recommends_the_smallest_candidate_near_the_best_fit():
    table = _table([
        {"min_leverage": 0.05, "n_selected": 40, "n_changed_mean": 38, "rmse_mean": 2.8},
        {"min_leverage": 0.03, "n_selected": 350, "n_changed_mean": 340, "rmse_mean": 1.40},
        {"min_leverage": 0.02, "n_selected": 1350, "n_changed_mean": 1330, "rmse_mean": 1.39},
        {"min_leverage": 0.005, "n_selected": 4400, "n_changed_mean": 4300, "rmse_mean": 1.38},
    ])
    cand, reason = recommend_tuning_settings(table)
    assert cand["min_leverage"] == 0.03
    assert cand["max_generations"] == 50          # 1.25 x plateau 40
    assert "fewest changed kcats" in reason


def test_a_candidate_whose_seeds_disagree_is_dropped():
    table = _table([
        {"min_leverage": 0.03, "n_selected": 350, "n_changed_mean": 340, "rmse_mean": 1.5},
        {"min_leverage": 0.005, "n_selected": 4400, "n_changed_mean": 4300,
         "rmse_mean": 1.0, "pct_direction_agree": 0.55},
    ])
    cand, reason = recommend_tuning_settings(table)
    assert cand["min_leverage"] == 0.03
    assert "dropped" in reason


def test_without_a_seed_pair_the_reproducibility_check_is_reported_as_skipped():
    table = _table([
        {"min_leverage": 0.03, "n_selected": 350, "n_changed_mean": 340,
         "rmse_mean": 1.5, "n_seeds": 1, "pct_direction_agree": np.nan,
         "rmse_sd": np.nan, "n_both_moved": 0},
    ])
    _, reason = recommend_tuning_settings(table)
    assert "could not be checked" in reason


def test_capped_candidate_does_not_get_a_generation_suggestion():
    table = _table([
        {"min_leverage": 0.03, "n_selected": 350, "n_changed_mean": 340,
         "rmse_mean": 1.5, "capped": True},
    ])
    cand, _ = recommend_tuning_settings(table)
    assert "max_generations" not in cand


def test_every_candidate_failing_the_check_raises():
    table = _table([
        {"min_leverage": 0.03, "n_selected": 350, "n_changed_mean": 340,
         "rmse_mean": 1.5, "pct_direction_agree": 0.4},
    ])
    with pytest.raises(ValueError, match="reproducib"):
        recommend_tuning_settings(table)


# --- search on the toy model -----------------------------------------------


def test_max_growth_weight_cannot_be_varied(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    params = EvotuneParams(source_groups=GROUPS, max_generations=5)
    with pytest.raises(ValueError, match="cannot be compared"):
        search_tuning_settings(
            model, [{"max_growth_weight": 2.0}], adapter=adapter, params=params,
            experiment_data=_experiment_data(), n_proc=1, verbose=False)


def test_unknown_setting_is_rejected(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    params = EvotuneParams(source_groups=GROUPS, max_generations=5)
    with pytest.raises(ValueError, match="Unknown setting"):
        search_tuning_settings(
            model, [{"colour": 1}], adapter=adapter, params=params,
            experiment_data=_experiment_data(), n_proc=1, verbose=False)


def test_search_runs_every_candidate_and_leaves_the_model_alone(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    before = model.ec.kcat.copy()
    experiment_data = _experiment_data()
    params = EvotuneParams(source_groups=GROUPS, max_generations=15, rmse_threshold=1e-6)
    screen = screen_kcat_leverage(
        model, adapter=adapter, params=params, experiment_data=experiment_data, n_proc=1)
    table, runs = search_tuning_settings(
        model, [{"target_impact_share": 1.0}, {"min_leverage": 0.0}],
        seeds=[0, 1], adapter=adapter, params=params, experiment_data=experiment_data,
        screen=screen, n_proc=1, verbose=False)
    assert len(runs) == 4
    assert np.array_equal(model.ec.kcat, before)
    assert len(table) == 2 and set(table["n_seeds"]) == {2}
    assert (table["n_selected"] == 2).all()
