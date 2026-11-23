"""format_kcat_source and parse_kcat_source, and the writers and readers of ec.source that use them."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import (
    KcatSource,
    apply_kcat_list,
    format_kcat_source,
    parse_kcat_source,
)
from geckopy.gather_kcats.kcat_source_format import is_standard_source

EXAMPLES = [
    "brenda (wc=0, origin=1)",
    "custom",
    "okp_catapro (similarity=42)",
    "combined (sd=0.61, n=6)",
    "brenda (wc=0, origin=2) | tuned=cmaes (prior=6.55)",
    "okp_catapro (similarity=42) | tuned=sensitivity+cmaes (prior=5.48)",
    "standard",
    "manual",
    "isozymes",
]


def _ec_model(rxns, source=None, kcat=None):
    n = len(rxns)
    model = EcModel("test")
    model.ec = EcData(
        rxns=list(rxns),
        kcat=np.array(kcat if kcat is not None else [0.0] * n, dtype=float),
        source=list(source if source is not None else [""] * n),
        notes=[""] * n,
        eccodes=[""] * n,
        rxn_enz_mat=sparse.csr_matrix((n, 0), dtype=float),
    )
    return model


# --------------------------------------------------------------------------- #
# format_kcat_source
# --------------------------------------------------------------------------- #

def test_format_each_example():
    assert format_kcat_source("brenda", {"wc": 0, "origin": 1}) == EXAMPLES[0]
    assert format_kcat_source("custom") == "custom"
    assert format_kcat_source("okp_catapro", {"similarity": 42}) == EXAMPLES[2]
    assert format_kcat_source("combined", {"sd": "0.61", "n": 6}) == EXAMPLES[3]
    assert format_kcat_source(
        "brenda", {"wc": 0, "origin": 2}, ["cmaes"], 6.55) == EXAMPLES[4]
    assert format_kcat_source(
        "okp_catapro", {"similarity": 42}, ["sensitivity", "cmaes"], 5.48) == EXAMPLES[5]
    assert format_kcat_source("standard") == "standard"
    assert format_kcat_source("manual") == "manual"
    assert format_kcat_source("isozymes") == "isozymes"


def test_format_normalises_origin_and_detail_keys():
    assert format_kcat_source("OKP-CataPro") == "okp_catapro"
    assert format_kcat_source("  Sabio--RK ") == "sabio_rk"
    assert format_kcat_source("brenda", {"WC-level": 1}) == "brenda (wc_level=1)"


def test_format_detail_values():
    assert format_kcat_source("a", {"x": 0.1234567}) == "a (x=0.123457)"
    assert format_kcat_source("a", {"x": 1234567.0}) == "a (x=1.23457e+06)"
    assert format_kcat_source("a", {"x": np.int64(3)}) == "a (x=3)"
    assert format_kcat_source("a", {"x": "0.60"}) == "a (x=0.60)"


def test_format_leaves_out_missing_detail_and_empty_brackets():
    assert format_kcat_source("brenda", {"wc": 0, "origin": None}) == "brenda (wc=0)"
    assert format_kcat_source("brenda", {"wc": math.nan, "origin": float("nan")}) == "brenda"
    assert format_kcat_source("brenda", {}) == "brenda"


def test_format_tuning_status_needs_methods():
    assert format_kcat_source("brenda", None, [], 5.0) == "brenda"
    assert format_kcat_source("brenda", None, ["cmaes"]) == "brenda | tuned=cmaes"
    assert format_kcat_source("brenda", None, ["cmaes"], math.nan) == "brenda | tuned=cmaes"
    assert format_kcat_source("brenda", None, iter(["sensitivity"]), 0.5) == (
        "brenda | tuned=sensitivity (prior=0.5)")


def test_format_empty_origin():
    assert format_kcat_source("") == ""
    assert format_kcat_source("", {}) == ""
    assert format_kcat_source("", {"wc": 1}) == "unknown (wc=1)"
    assert format_kcat_source("", None, ["cmaes"], 2.0) == "unknown | tuned=cmaes (prior=2)"


# --------------------------------------------------------------------------- #
# parse_kcat_source
# --------------------------------------------------------------------------- #

def test_parse_each_example():
    assert parse_kcat_source(EXAMPLES[0]) == KcatSource("brenda", {"wc": 0.0, "origin": 1.0})
    assert parse_kcat_source("custom") == KcatSource("custom", {}, (), None)
    assert parse_kcat_source(EXAMPLES[2]) == KcatSource("okp_catapro", {"similarity": 42.0})
    assert parse_kcat_source(EXAMPLES[3]) == KcatSource("combined", {"sd": 0.61, "n": 6.0})
    assert parse_kcat_source(EXAMPLES[4]) == KcatSource(
        "brenda", {"wc": 0.0, "origin": 2.0}, ("cmaes",), 6.55)
    assert parse_kcat_source(EXAMPLES[5]) == KcatSource(
        "okp_catapro", {"similarity": 42.0}, ("sensitivity", "cmaes"), 5.48)


def test_parse_detail_text_and_odd_parts():
    info = parse_kcat_source("x (note=abc, junk, wc = 2 ,=3)")
    assert info.origin == "x"
    assert info.detail == {"note": "abc", "wc": 2.0}


def test_parse_status_without_prior():
    info = parse_kcat_source("brenda | tuned=sensitivity+cmaes")
    assert info.tuned == ("sensitivity", "cmaes")
    assert info.prior is None


@pytest.mark.parametrize("text, origin", [
    ("brenda", "brenda"),
    ("OKP-CataPro", "OKP-CataPro"),
    ("sensitivityTuning", "sensitivityTuning"),
    ("  Custom ", "Custom"),
    ("setKcatForReactions", "setKcatForReactions"),
])
def test_parse_legacy_strings_are_origins_as_written(text, origin):
    assert parse_kcat_source(text) == KcatSource(origin, {}, (), None)


@pytest.mark.parametrize("value", ["", None, math.nan, np.nan, 3, 2.5, ["brenda"]])
def test_parse_missing_or_non_text_gives_empty_origin(value):
    assert parse_kcat_source(value) == KcatSource("", {}, (), None)


def test_parse_result_is_frozen():
    with pytest.raises(AttributeError):
        parse_kcat_source("brenda").origin = "x"


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", EXAMPLES)
def test_format_of_parse_returns_the_string(text):
    info = parse_kcat_source(text)
    assert format_kcat_source(info.origin, info.detail, info.tuned, info.prior) == text


def test_round_trip_changes_only_the_number_format():
    info = parse_kcat_source("combined (sd=0.60, n=6)")
    assert format_kcat_source(info.origin, info.detail) == "combined (sd=0.6, n=6)"


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def test_apply_kcat_list_writes_the_strings_it_always_wrote():
    model = _ec_model(["r1", "r2", "r3", "r4"])
    kcat_list = pd.DataFrame({
        "rxn_id": ["r1", "r2", "r3", "r4"],
        "kcat": [1.0, 2.0, 3.0, 4.0],
        "source": ["BRENDA", "DLKcat", "OKP-CataPro", "Sabio-RK"],
        "wildcard_level": [0, 2, np.nan, 1],
        "origin": [1, np.nan, np.nan, 3],
    })
    apply_kcat_list(model, kcat_list)
    assert model.ec.source == [
        "brenda (wc=0, origin=1)",
        "dlkcat (wc=2)",
        "okp_catapro",
        "sabio_rk (wc=1, origin=3)",
    ]


def test_apply_kcat_list_without_fuzzy_columns_writes_the_bare_token():
    model = _ec_model(["r1"])
    apply_kcat_list(model, pd.DataFrame({"rxn_id": ["r1"], "kcat": [1.0], "source": ["DLKcat"]}))
    assert model.ec.source == ["dlkcat"]


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source, expected", [
    ("standard", True),
    ("standard | tuned=sensitivity (prior=0.5)", False),
    ("standard | tuned=cmaes (prior=0.5)", False),
    ("Standard", False),
    ("standardised", False),
    ("brenda", False),
    ("", False),
])
def test_standard_source_needs_the_origin_and_no_tuning_status(source, expected):
    assert is_standard_source(source) is expected


def test_parse_reads_an_unclosed_bracket_as_one_origin():
    info = parse_kcat_source("brenda (wc=1")
    assert info.origin == "brenda (wc=1"
    assert info.detail == {}
    assert info.tuned == ()
    assert info.prior is None


def test_parse_reads_a_number_only_in_plain_decimal_form():
    info = parse_kcat_source("x (a=1_0, b=inf, c=1e3, d=.5, e=-2., f=0x10)")
    assert info.detail == {
        "a": "1_0", "b": "inf", "c": 1000.0, "d": 0.5, "e": -2.0, "f": "0x10",
    }


def test_parse_ignores_a_pair_whose_key_does_not_start_with_a_letter():
    info = parse_kcat_source("x (1a=1, a b=2, ok=3, _c=4)")
    assert info.detail == {"ok": 3.0}


def test_parse_needs_one_space_before_the_prior():
    info = parse_kcat_source("x | tuned=cmaes(prior=3)")
    assert info.tuned == ("cmaes(prior=3)",)
    assert info.prior is None


def test_format_leaves_out_pairs_without_a_usable_key_or_value():
    assert format_kcat_source("x", {"": 1, "1a": 2, "ok": 3}) == "x (ok=3)"
    assert format_kcat_source("x", {"a": float("inf"), "b": 1}) == "x (b=1)"
    assert format_kcat_source("x", {"a": float("-inf")}) == "x"


def test_format_leaves_out_a_prior_that_is_not_finite():
    assert format_kcat_source("x", None, ["cmaes"], float("inf")) == "x | tuned=cmaes"


def test_format_takes_a_single_method_as_text():
    assert format_kcat_source("x", None, "cmaes", 2) == "x | tuned=cmaes (prior=2)"


def test_parse_needs_exactly_one_space_before_the_prior():
    info = parse_kcat_source("x | tuned=cmaes  (prior=3)")
    assert info.tuned == ("cmaes  (prior=3)",)
    assert info.prior is None


def test_parse_reads_a_closing_bracket_inside_the_prior_as_a_method():
    info = parse_kcat_source("x | tuned=cmaes (prior=1))")
    assert info.tuned == ("cmaes (prior=1))",)
    assert info.prior is None


def test_parse_reads_a_number_of_ascii_digits_only():
    info = parse_kcat_source("x (a=\u0663, b=3)")
    assert info.detail == {"a": "\u0663", "b": 3.0}


def test_format_takes_no_methods_as_none():
    assert format_kcat_source("x", None, None, 2) == "x"


def test_parse_trims_the_head_before_the_tuning_status():
    info = parse_kcat_source("x (a=1)  | tuned=cmaes")
    assert info.origin == "x"
    assert info.detail == {"a": 1.0}
    assert info.tuned == ("cmaes",)


def test_parse_takes_the_first_prior_marker():
    info = parse_kcat_source("x | tuned=cmaes (prior=3) (prior=4)")
    assert info.tuned == ("cmaes (prior=3) (prior=4)",)
    assert info.prior is None

