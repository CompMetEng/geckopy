"""Tests for kcat_tuning.evotune.parsimony."""
import numpy as np
import pytest

from geckopy.kcat_tuning.evotune.parsimony import (
    best_parsimonious,
    movement_in_sigma,
    n_changed,
    parsimony_frontier,
    revert_below,
)


def test_movement_is_scaled_by_each_kcats_own_sigma():
    """The same fractional change counts more for a trusted kcat."""
    kcat0 = np.array([10.0, 10.0])
    kcat = kcat0 * np.exp(0.3)
    sigma0 = np.array([0.1, 0.3])          # custom-like, unlabelled-like
    dev = movement_in_sigma(kcat, kcat0, sigma0)
    assert dev == pytest.approx([3.0, 1.0])


def test_revert_below_keeps_only_the_large_movers():
    kcat0 = np.array([1.0, 1.0, 1.0])
    sigma0 = np.array([0.1, 0.1, 0.1])
    kcat = kcat0 * np.exp([0.05, 0.2, 0.5])     # 0.5, 2.0, 5.0 sigma
    out = revert_below(kcat, kcat0, sigma0, threshold=1.0)
    assert out[0] == pytest.approx(kcat0[0])    # reverted
    assert out[1] == pytest.approx(kcat[1])     # kept
    assert out[2] == pytest.approx(kcat[2])     # kept
    # A zero threshold reverts nothing.
    assert revert_below(kcat, kcat0, sigma0, 0.0) == pytest.approx(kcat)


def test_n_changed_uses_the_two_percent_convention():
    kcat0 = np.array([1.0, 1.0, 1.0])
    kcat = np.array([1.0, 1.01, 1.5])           # unchanged, under 2%, over
    assert n_changed(kcat, kcat0) == 1


def test_frontier_reverts_progressively_and_scores_each_point():
    kcat0 = np.ones(4)
    sigma0 = np.full(4, 0.1)
    kcat = kcat0 * np.exp([0.05, 0.15, 0.35, 0.55])   # 0.5, 1.5, 3.5, 5.5 sigma
    calls = []

    def score(vec):
        calls.append(vec.copy())
        return float(np.abs(np.log(vec / kcat0)).sum())

    pts = parsimony_frontier(score, kcat, kcat0, sigma0,
                             thresholds=(0.0, 2.0, 4.0, 6.0))
    assert len(pts) == len(calls) == 4
    # Reverting more can only reduce the number of changed kcats.
    assert [p.n_changed for p in pts] == sorted(
        [p.n_changed for p in pts], reverse=True
    )
    assert pts[0].n_changed == 4 and pts[-1].n_changed == 0
    assert pts[0].threshold == 0.0


def test_best_parsimonious_prefers_fewer_changes_within_tolerance():
    from geckopy.kcat_tuning.evotune.parsimony import FrontierPoint

    pts = [
        FrontierPoint(0.0, np.array([1.0]), 4000, 7.0, 0.900),
        FrontierPoint(3.0, np.array([1.0]), 3000, 6.0, 0.905),   # within 2%
        FrontierPoint(6.0, np.array([1.0]), 1000, 3.0, 1.400),   # too costly
    ]
    assert best_parsimonious(pts, tolerance=0.02).n_changed == 3000
    # A tight tolerance falls back to the best-fitting point.
    assert best_parsimonious(pts, tolerance=0.0).n_changed == 4000


def test_best_parsimonious_skips_nan_rmse():
    from geckopy.kcat_tuning.evotune.parsimony import FrontierPoint

    pts = [
        FrontierPoint(0.0, np.array([1.0]), 4000, 7.0, float("nan")),
        FrontierPoint(3.0, np.array([1.0]), 3000, 6.0, 0.905),
        FrontierPoint(6.0, np.array([1.0]), 1000, 3.0, 1.400),
    ]
    assert best_parsimonious(pts).n_changed == 3000
    nan_only = [FrontierPoint(0.0, np.array([1.0]), 1, 1.0, float("nan"))]
    with pytest.raises(ValueError):
        best_parsimonious(nan_only)


def test_fold_change_is_symmetric_and_at_least_one():
    from geckopy.kcat_tuning.evotune.parsimony import fold_change

    kcat0 = np.array([10.0, 10.0, 10.0])
    kcat = np.array([20.0, 5.0, 10.0])       # doubled, halved, unchanged
    assert fold_change(kcat, kcat0) == pytest.approx([2.0, 2.0, 1.0])


def test_source_movement_flags_whether_trust_order_is_respected():
    from geckopy.kcat_tuning.evotune.parsimony import source_movement

    kcat0 = np.ones(6)
    groups = np.array(["custom", "custom", "brenda", "brenda", "okp", "okp"])
    order = ["custom", "brenda", "okp"]

    # Trusted sources moved least: the ordering holds.
    good = np.array([1.05, 1.05, 1.5, 1.5, 3.0, 3.0])
    rep = source_movement(good, kcat0, groups, order)
    assert rep["_ordered"] is True
    assert rep["custom"]["median_fold"] == pytest.approx(1.05)
    assert rep["custom"]["frac_near_prior"] == pytest.approx(1.0)
    assert rep["okp"]["frac_near_prior"] == pytest.approx(0.0)

    # The most trusted source moved most: flagged.
    bad = np.array([3.0, 3.0, 1.5, 1.5, 1.05, 1.05])
    assert source_movement(bad, kcat0, groups, order)["_ordered"] is False


def test_source_movement_reads_the_tier_when_most_kcats_are_untouched():
    from geckopy.kcat_tuning.evotune.parsimony import source_movement

    kcat0 = np.ones(10)
    groups = np.array(["custom"] * 5 + ["brenda"] * 5)
    order = ["custom", "brenda"]

    # One kcat moved per source, custom four-fold against brenda's two.
    kcat = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    rep = source_movement(kcat, kcat0, groups, order)

    # Over the whole source both medians sit at the prior and say
    # nothing; conditioning on what moved shows custom moving furthest.
    assert rep["custom"]["median_fold"] == pytest.approx(1.0)
    assert rep["brenda"]["median_fold"] == pytest.approx(1.0)
    assert rep["custom"]["median_fold_moved"] == pytest.approx(4.0)
    assert rep["brenda"]["median_fold_moved"] == pytest.approx(2.0)
    assert rep["custom"]["n_moved"] == 1
    assert rep["_ordered"] is False

    # A source with nothing moved reports 1.0 and does not decide the
    # ordering on its own.
    still = np.array([1.0] * 5 + [1.0, 1.0, 1.0, 1.0, 2.0])
    rep = source_movement(still, kcat0, groups, order)
    assert rep["custom"]["n_moved"] == 0
    assert rep["custom"]["median_fold_moved"] == pytest.approx(1.0)
    assert rep["_ordered"] is True


def test_identifiability_mask_asks_more_of_trusted_sources():
    from geckopy.kcat_tuning.evotune.parsimony import identifiability_mask

    # Identical measured effect, different trust: custom (0.1) must clear
    # a bar three times higher than unlabelled (0.3).
    drmse = np.array([1e-3, 1e-3])
    sigma0 = np.array([0.1, 0.3])
    assert list(identifiability_mask(drmse, sigma0, 1.5e-4)) == [False, True]
    # A large enough effect qualifies whatever the source.
    assert list(identifiability_mask(np.array([1e-2, 1e-2]), sigma0, 1.5e-4)) == [True, True]
    # And nothing qualifies under an unreachable bar.
    assert not identifiability_mask(drmse, sigma0, 1.0).any()


def test_impact_share_rewards_changing_the_kcats_that_matter():
    from geckopy.kcat_tuning.evotune.parsimony import impact_share

    kcat0 = np.ones(4)
    drmse = np.array([1.0, 1.0, 0.01, 0.01])     # two matter, two barely

    # Two changes, both load-bearing.
    few_good = np.array([2.0, 2.0, 1.0, 1.0])
    assert impact_share(few_good, kcat0, drmse) == pytest.approx(2.0 / 2.02)

    # Two changes, neither load-bearing: same count, far less impact.
    few_bad = np.array([1.0, 1.0, 2.0, 2.0])
    assert impact_share(few_bad, kcat0, drmse) == pytest.approx(0.02 / 2.02)
    assert impact_share(few_bad, kcat0, drmse) < impact_share(few_good, kcat0, drmse)

    # Nothing changed carries nothing.
    assert impact_share(kcat0, kcat0, drmse) == 0.0
