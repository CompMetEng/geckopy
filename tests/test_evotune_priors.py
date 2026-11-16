"""Tests for kcat_tuning.evotune.priors."""
import numpy as np
import pytest

from geckopy.adapter.params import EvotuneParams, SourceGroupRule
from geckopy.kcat_tuning.evotune.priors import (
    UNLABELLED_GROUP,
    build_sigma0_log,
    classify_kcat_source,
    classify_kcat_sources,
    warn_unlabelled,
)


def _params() -> EvotuneParams:
    return EvotuneParams(
        sigma0_log_default=0.5,
        source_groups={
            "dlkcat": SourceGroupRule(sources=["dlkcat"], sigma0_log=0.4),
            "brenda": SourceGroupRule(sources=["brenda"], sigma0_log=0.2),
            "okp": SourceGroupRule(sources=["okp_*"], sigma0_log=0.3),
        },
    )


# --------------------------------------------------------------------------- #
# classify_kcat_source(s)
# --------------------------------------------------------------------------- #

def test_classify_matches_explicit_source_case_insensitively():
    params = _params()
    assert classify_kcat_source("DLKcat", params) == "dlkcat"
    assert classify_kcat_source("brenda", params) == "brenda"
    assert classify_kcat_source("BRENDA", params) == "brenda"


def test_classify_unmatched_source_is_unlabelled():
    params = _params()
    assert classify_kcat_source("some_other_source", params) == UNLABELLED_GROUP


def test_classify_matches_prefix_pattern_on_the_origin():
    params = _params()
    assert classify_kcat_source("okp_kinform_h", params) == "okp"
    assert classify_kcat_source("OKP_EITLEM_Kinetics (similarity=0.9)", params) == "okp"
    assert classify_kcat_source("okp", params) == UNLABELLED_GROUP
    assert classify_kcat_source("xokp_kinform_h", params) == UNLABELLED_GROUP


def test_exact_matches_win_over_patterns_and_longest_prefix_wins():
    params = EvotuneParams(source_groups={
        "wide": SourceGroupRule(sources=["okp_*"], sigma0_log=0.3),
        "narrow": SourceGroupRule(sources=["okp_kin*"], sigma0_log=0.2),
        "same": SourceGroupRule(sources=["okp_kin*"], sigma0_log=0.25),
        "exact": SourceGroupRule(sources=["okp_kinform_h"], sigma0_log=0.1),
    })
    assert classify_kcat_source("okp_kinform_h", params) == "exact"  # exact over every pattern
    assert classify_kcat_source("okp_kinform_l", params) == "narrow"  # longest prefix; ties keep group order
    assert classify_kcat_source("okp_catapro", params) == "wide"


def test_pattern_group_listed_first_does_not_beat_an_exact_match_of_a_later_group():
    params = EvotuneParams(source_groups={
        "all": SourceGroupRule(sources=["*"], sigma0_log=0.5),
        "brenda": SourceGroupRule(sources=["brenda"], sigma0_log=0.2),
    })
    assert classify_kcat_source("brenda", params) == "brenda"
    assert classify_kcat_source("anything", params) == "all"


def test_classify_kcat_sources_vectorised():
    params = _params()
    groups = classify_kcat_sources(["dlkcat", "brenda", "okp_unikp", "unknown"], params)
    assert list(groups) == ["dlkcat", "brenda", "okp", UNLABELLED_GROUP]


def test_unlabelled_warning_names_count_default_and_common_origins(caplog):
    params = _params()
    sources = ["brenda"] * 5 + ["standard"] * 3 + ["isozymes"] * 2 + ["a"] + ["b"] + ["c"] + ["d"]
    groups = classify_kcat_sources(sources, params)
    with caplog.at_level("WARNING"), pytest.warns(UserWarning, match="match no evotune source group"):
        warn_unlabelled(groups, sources, params)
    assert (
        "9 of 14 tunable kcats match no evotune source group and get sigma0_log_default (0.5); "
        "origins: standard, isozymes, a, b, c"
    ) in caplog.text


def test_unlabelled_warning_only_above_a_fifth(caplog):
    params = _params()
    sources = ["brenda"] * 4 + ["standard"]  # exactly 20 %
    with caplog.at_level("WARNING"):
        warn_unlabelled(classify_kcat_sources(sources, params), sources, params)
    assert "match no evotune source group" not in caplog.text


# --------------------------------------------------------------------------- #
# build_sigma0_log
# --------------------------------------------------------------------------- #

def test_build_sigma0_log_uses_group_values_and_default_fallback():
    params = _params()
    groups = np.array(["dlkcat", "brenda", UNLABELLED_GROUP, "okp"], dtype=object)

    sigma0_log = build_sigma0_log(groups, params)

    assert sigma0_log.tolist() == pytest.approx([0.4, 0.2, 0.5, 0.3])


def test_a_tuning_status_does_not_hide_the_source():
    params = EvotuneParams(
        source_groups={"brenda": SourceGroupRule(sources=["brenda"], sigma0_log=0.2)},
        sigma0_log_default=0.3,
    )
    for source in ("brenda | tuned=cmaes (prior=6.55)",
                   "BRENDA (wc=0, origin=2) | tuned=sensitivity+cmaes (prior=5.48)"):
        assert classify_kcat_source(source, params) == "brenda"
    assert classify_kcat_source("brendax | tuned=cmaes (prior=1)", params) == UNLABELLED_GROUP


def test_a_fuzzy_match_detail_does_not_hide_the_source():
    """``ec.source`` carries the wildcard/origin detail; the tier is the
    source token in front of it."""
    params = EvotuneParams(
        source_groups={"brenda": SourceGroupRule(sources=["brenda"], sigma0_log=0.2)},
        sigma0_log_default=0.3,
    )
    assert classify_kcat_source("brenda (wc=0, origin=1)", params) == "brenda"
    assert classify_kcat_source("brenda", params) == "brenda"
    assert classify_kcat_source("dlkcat", params) == UNLABELLED_GROUP
    widths = build_sigma0_log(
        classify_kcat_sources(
            ["brenda (wc=1, origin=3)", "dlkcat"], params,
        ),
        params,
    )
    assert widths.tolist() == [0.2, 0.3]


def test_a_group_may_still_list_the_whole_source_string():
    params = EvotuneParams(
        source_groups={
            "wildcarded": SourceGroupRule(sources=["brenda (wc=1, origin=3)"], sigma0_log=0.5),
        },
    )
    assert classify_kcat_source(
        "brenda (wc=1, origin=3)", params,
    ) == "wildcarded"
