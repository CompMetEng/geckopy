"""Tests for kcat_tuning.evotune.priors."""
import numpy as np
import pytest

from geckopy.adapter.params import EvotuneParams, SourceGroupRule
from geckopy.kcat_tuning.evotune.priors import (
    UNLABELLED_GROUP,
    build_sigma0_log,
    classify_kcat_source,
    classify_kcat_sources,
)


def _params() -> EvotuneParams:
    return EvotuneParams(
        sigma0_log_default=0.5,
        source_groups={
            "dlkcat": SourceGroupRule(sources=["dlkcat"]),
            "brenda": SourceGroupRule(sources=["brenda"]),
            "okp": SourceGroupRule(match_okp=True),
        },
        sigma0_log_source={"dlkcat": 0.4, "brenda": 0.2, "okp": 0.3},
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


def test_classify_matches_okp_method():
    params = _params()
    assert classify_kcat_source("CataPro", params, okp_method="CataPro") == "okp"
    # Case-insensitive here too.
    assert classify_kcat_source("catapro", params, okp_method="CataPro") == "okp"
    # No okp_method configured -> falls through to unlabelled.
    assert classify_kcat_source("CataPro", params, okp_method=None) == UNLABELLED_GROUP


def test_classify_kcat_sources_vectorised():
    params = _params()
    groups = classify_kcat_sources(
        ["dlkcat", "brenda", "CataPro", "unknown"], params, okp_method="CataPro",
    )
    assert list(groups) == ["dlkcat", "brenda", "okp", UNLABELLED_GROUP]


# --------------------------------------------------------------------------- #
# build_sigma0_log
# --------------------------------------------------------------------------- #

def test_build_sigma0_log_uses_group_values_and_default_fallback():
    params = _params()
    groups = np.array(["dlkcat", "brenda", UNLABELLED_GROUP, "okp"], dtype=object)

    sigma0_log = build_sigma0_log(groups, params)

    assert sigma0_log.tolist() == pytest.approx([0.4, 0.2, 0.5, 0.3])


def test_a_fuzzy_match_detail_does_not_hide_the_source():
    """``ec.source`` carries the wildcard/origin detail; the tier is the
    source token in front of it."""
    params = EvotuneParams(
        source_groups={"brenda": SourceGroupRule(sources=["brenda"])},
        sigma0_log_source={"brenda": 0.2},
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
            "wildcarded": SourceGroupRule(sources=["brenda (wc=1, origin=3)"]),
        },
        sigma0_log_source={"wildcarded": 0.5},
    )
    assert classify_kcat_source(
        "brenda (wc=1, origin=3)", params,
    ) == "wildcarded"
