"""Behaviour that follows GECKO: truncate_values ties, the detail text of a kcat source file,
and the defaults of fill_kcats_from_isozymes."""
import numpy as np
import pandas as pd
import pytest

from geckopy.ec_model.pipeline import fill_kcats_from_isozymes, set_kcat_for_reactions
from geckopy.gather_kcats import read_kcat_source, save_kcat_source
from geckopy.kcat_tuning import truncate_values
from geckopy.kcat_tuning.truncate_values import _truncate_scalar

from test_fill_kcats import _ectestgem_ec_model, _isozyme_model, _kcat
from test_kcat_sources import _adapter, _kcat_list, _model


# --------------------------------------------------------------------------- #
# truncate_values: a tie rounds away from zero, as MATLAB round(x, n) does
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value, expected", [
    (100000.5, 100001.0),
    (1234565.0, 1234570.0),
    (-100000.5, -100001.0),
    (0.1234565, 0.123457),
])
def test_truncate_tie_rounds_away_from_zero(value, expected):
    assert _truncate_scalar(value) == expected


def test_truncate_array_ties_and_unchanged_values():
    out = truncate_values(np.array([100000.5, 1234565.0, 0.0, np.inf, np.nan, 12.3456789]))
    assert out[0] == 100001.0 and out[1] == 1234570.0
    assert out[2] == 0.0 and np.isinf(out[3]) and np.isnan(out[4])
    assert out[5] == pytest.approx(12.3457)


# --------------------------------------------------------------------------- #
# save_kcat_source: the detail column can not split a row
# --------------------------------------------------------------------------- #

def _saved_details(tmp_path, extras):
    adapter = _adapter(tmp_path)
    model = _model(adapter, light=False)
    kcats = _kcat_list(model)
    kcats["detail"] = [extras[i % len(extras)] for i in range(len(kcats))]
    path = save_kcat_source(model, kcats, "brenda")
    data_lines = [line for line in path.read_text(encoding="utf-8").split("\n")
                  if line and not line.startswith("#")]
    return data_lines, read_kcat_source(path)[0]


def test_detail_with_tab_and_newline_keeps_one_row_per_kcat(tmp_path):
    lines, records = _saved_details(tmp_path, ["a\tb", "c\nd"])
    assert {len(line.split("\t")) for line in lines} == {9}
    assert len(records) == len(lines) - 1
    assert set(records["detail"]) <= {"wc=0; origin=%d; a b" % o for o in range(1, 7)} | {
        "wc=0; origin=%d; c d" % o for o in range(1, 7)}


def test_detail_leaves_no_empty_parts(tmp_path):
    _, records = _saved_details(tmp_path, ["; x", "p; ; q"])
    details = set(records["detail"])
    assert all("; ;" not in d and not d.endswith("; ") for d in details)
    assert "wc=0; origin=1; x" in details
    assert "wc=0; origin=2; p; q" in details


# --------------------------------------------------------------------------- #
# fill_kcats_from_isozymes: only a kcat of 0 is missing; apply is off by default
# --------------------------------------------------------------------------- #

def test_nan_kcat_is_not_filled_from_siblings():
    model = _isozyme_model([5.0, float("nan"), 0.0])
    fill_kcats_from_isozymes(model)
    assert np.isnan(model.ec.kcat[1])
    assert model.ec.source[1] == "x"
    assert model.ec.kcat[2] == 5.0 and model.ec.source[2] == "isozymes"


def test_nan_sibling_does_not_enter_the_mean():
    model = _isozyme_model([4.0, float("nan"), 8.0, 0.0])
    fill_kcats_from_isozymes(model)
    assert model.ec.kcat[3] == 6.0


def test_apply_is_off_by_default():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)
    fill_kcats_from_isozymes(ec_model)
    assert _kcat(ec_model, "R2_EXP_2") == 100.0
    r = ec_model.reactions.get_by_id("R2_EXP_2")
    assert next((c for m, c in r.metabolites.items() if m.id == "prot_P3"), 0.0) == 0.0
