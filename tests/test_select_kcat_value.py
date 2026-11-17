"""Tests for apply_kcat_list."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import apply_kcat_list


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _ec_model(
    ec_rxns: list[str],
    *,
    initial_kcat: list[float] | None = None,
    initial_source: list[str] | None = None,
) -> EcModel:
    """Build an EcModel with the given ec.rxns; ``initial_kcat`` and
    ``initial_source`` default to all-zero and all-empty-string when
    omitted."""
    n = len(ec_rxns)
    if initial_kcat is None:
        initial_kcat = [0.0] * n
    if initial_source is None:
        initial_source = [""] * n
    model = EcModel("test")
    model.ec = EcData(
        rxns=list(ec_rxns),
        kcat=np.array(initial_kcat, dtype=float),
        source=list(initial_source),
        notes=[""] * n,
        eccodes=[""] * n,
        rxn_enz_mat=sparse.csr_matrix((n, 0), dtype=float),
    )
    return model


def _kcat_list(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Build a kcat_list DataFrame from (rxn_id, kcat, source) tuples."""
    return pd.DataFrame(rows, columns=["rxn_id", "kcat", "source"])


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_empty_kcat_list_returns_empty_no_updates():
    model = _ec_model(["r1"])
    updated = apply_kcat_list(model, _kcat_list([]))
    assert updated == []
    assert model.ec.kcat[0] == 0


def test_all_zero_kcats_no_updates():
    model = _ec_model(["r1"])
    updated = apply_kcat_list(
        model, _kcat_list([("r1", 0.0, "brenda")]),
    )
    assert updated == []
    assert model.ec.kcat[0] == 0


def test_single_kcat_per_reaction_basic():
    model = _ec_model(["r1", "r2"])
    updated = apply_kcat_list(
        model,
        _kcat_list([("r1", 5.0, "brenda"), ("r2", 7.0, "brenda")]),
    )
    assert sorted(updated) == ["r1", "r2"]
    np.testing.assert_array_equal(model.ec.kcat, [5.0, 7.0])
    assert model.ec.source == ["brenda", "brenda"]


def test_only_reactions_in_list_are_updated():
    model = _ec_model(["r1", "r2", "r3"], initial_kcat=[1.0, 1.0, 1.0])
    apply_kcat_list(model, _kcat_list([("r2", 5.0, "brenda")]))
    np.testing.assert_array_equal(model.ec.kcat, [1.0, 5.0, 1.0])


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #

def test_criteria_max_picks_largest():
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 1.0, "src1"),
            ("r1", 5.0, "src2"),
            ("r1", 2.0, "src3"),
        ]),
        criteria="max",
    )
    assert model.ec.kcat[0] == 5.0
    assert model.ec.source[0] == "src2"


def test_criteria_min_picks_smallest():
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 5.0, "src1"),
            ("r1", 1.0, "src2"),
            ("r1", 2.0, "src3"),
        ]),
        criteria="min",
    )
    assert model.ec.kcat[0] == 1.0
    assert model.ec.source[0] == "src2"


def test_criteria_median_picks_median_value():
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 1.0, "src_first"),
            ("r1", 3.0, "src_middle"),
            ("r1", 5.0, "src_last"),
        ]),
        criteria="median",
    )
    assert model.ec.kcat[0] == 3.0


def test_criteria_median_attributes_source_to_first_row():
    """For median/mean, the recorded source comes from the first row of
    the group, not from the row whose kcat is the median sample itself."""
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 1.0, "src_first"),
            ("r1", 3.0, "src_middle"),
            ("r1", 5.0, "src_last"),
        ]),
        criteria="median",
    )
    assert model.ec.source[0] == "src_first"


def test_criteria_mean_picks_average():
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 2.0, "a"),
            ("r1", 4.0, "b"),
            ("r1", 6.0, "c"),
        ]),
        criteria="mean",
    )
    assert model.ec.kcat[0] == pytest.approx(4.0)


def test_criteria_mean_attributes_source_to_first_row():
    model = _ec_model(["r1"])
    apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 2.0, "src_first"),
            ("r1", 4.0, "src_second"),
        ]),
        criteria="mean",
    )
    assert model.ec.source[0] == "src_first"


def test_criteria_invalid_raises():
    model = _ec_model(["r1"])
    with pytest.raises(ValueError, match="criteria must be"):
        apply_kcat_list(
            model, _kcat_list([("r1", 5.0, "a")]), criteria="nonsense",
        )


# --------------------------------------------------------------------------- #
# overwrite modes
# --------------------------------------------------------------------------- #

def test_overwrite_true_replaces_existing():
    model = _ec_model(["r1"], initial_kcat=[10.0], initial_source=["old"])
    updated = apply_kcat_list(
        model, _kcat_list([("r1", 5.0, "new")]), overwrite=True,
    )
    assert updated == ["r1"]
    assert model.ec.kcat[0] == 5.0
    assert model.ec.source[0] == "new"


def test_overwrite_false_only_updates_unset():
    model = _ec_model(
        ["r1", "r2"],
        initial_kcat=[10.0, 0.0],
        initial_source=["old1", "old2"],
    )
    updated = apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 100.0, "new1"),
            ("r2", 200.0, "new2"),
        ]),
        overwrite=False,
    )
    assert updated == ["r2"]
    assert model.ec.kcat[0] == 10.0  # untouched
    assert model.ec.kcat[1] == 200.0


def test_overwrite_if_higher_only_replaces_when_strictly_higher():
    model = _ec_model(
        ["r1", "r2", "r3"],
        initial_kcat=[10.0, 10.0, 10.0],
        initial_source=["old1", "old2", "old3"],
    )
    updated = apply_kcat_list(
        model,
        _kcat_list([
            ("r1", 5.0, "lower"),    # 5 < 10 -> skip
            ("r2", 10.0, "equal"),   # 10 == 10 -> skip
            ("r3", 15.0, "higher"),  # 15 > 10 -> update
        ]),
        overwrite="if_higher",
    )
    assert updated == ["r3"]
    assert model.ec.kcat[0] == 10.0
    assert model.ec.kcat[1] == 10.0
    assert model.ec.kcat[2] == 15.0


def test_overwrite_if_higher_treats_zero_as_unset():
    model = _ec_model(
        ["r1"],
        initial_kcat=[0.0],
        initial_source=["old1"],
    )
    updated = apply_kcat_list(
        model,
        _kcat_list([("r1", 5.0, "new1")]),
        overwrite="if_higher",
    )
    assert updated == ["r1"]
    assert model.ec.kcat[0] == 5.0


def test_overwrite_false_treats_nan_as_unset():
    model = _ec_model(["r1"], initial_kcat=[float("nan")], initial_source=["old"])
    updated = apply_kcat_list(
        model, _kcat_list([("r1", 5.0, "new")]), overwrite=False,
    )
    assert updated == ["r1"]
    assert model.ec.kcat[0] == 5.0


def test_overwrite_if_higher_treats_nan_as_unset():
    model = _ec_model(["r1"], initial_kcat=[float("nan")], initial_source=["old"])
    updated = apply_kcat_list(
        model, _kcat_list([("r1", 5.0, "new")]), overwrite="if_higher",
    )
    assert updated == ["r1"]
    assert model.ec.kcat[0] == 5.0


def test_overwrite_invalid_raises():
    model = _ec_model(["r1"])
    with pytest.raises(ValueError, match="overwrite must be"):
        apply_kcat_list(
            model, _kcat_list([("r1", 5.0, "a")]), overwrite="nonsense",
        )


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def test_unknown_rxn_id_raises():
    model = _ec_model(["r1"])
    with pytest.raises(ValueError, match="not present in model.ec.rxns"):
        apply_kcat_list(model, _kcat_list([("nonexistent", 5.0, "a")]))


def test_missing_column_raises():
    model = _ec_model(["r1"])
    bad_list = pd.DataFrame({"rxn_id": ["r1"], "kcat": [5.0]})  # no source
    with pytest.raises(KeyError, match="source"):
        apply_kcat_list(model, bad_list)


# --------------------------------------------------------------------------- #
# Return value
# --------------------------------------------------------------------------- #

def test_return_value_is_list_of_rxn_ids():
    model = _ec_model(["r1", "r2"])
    updated = apply_kcat_list(
        model,
        _kcat_list([("r1", 5.0, "a"), ("r2", 7.0, "b")]),
    )
    assert isinstance(updated, list)
    assert all(isinstance(r, str) for r in updated)
    assert sorted(updated) == ["r1", "r2"]


def test_return_value_omits_reactions_not_updated_under_if_higher():
    model = _ec_model(["r1"], initial_kcat=[10.0])
    updated = apply_kcat_list(
        model, _kcat_list([("r1", 5.0, "lower")]), overwrite="if_higher",
    )
    assert updated == []


# --------------------------------------------------------------------------- #
# Integration with fuzzy_kcat_matching DataFrame schema
# --------------------------------------------------------------------------- #

def test_consumes_fuzzy_kcat_matching_schema_directly():
    """The full schema produced by fuzzy_kcat_matching should pass
    through without column adjustments."""
    model = _ec_model(["r1", "r2"])
    df = pd.DataFrame({
        "rxn_id": ["r1", "r2", "r1"],
        "source": ["brenda", "brenda", "brenda"],
        "eccode": ["1.1.1.1", "2.7.7.7", "1.1.1.1"],
        "substrates": [["A"], ["B"], ["A"]],
        "genes": [[], [], []],
        "kcat": [5.0, 7.0, 9.0],
        "wildcard_level": pd.array([0, 1, 0], dtype="Int64"),
        "origin": pd.array([1, 3, 1], dtype="Int64"),
    })
    updated = apply_kcat_list(model, df, criteria="max")
    assert sorted(updated) == ["r1", "r2"]
    # r1 had two entries (5.0, 9.0); max is 9.0.
    assert model.ec.kcat[0] == 9.0
    assert model.ec.kcat[1] == 7.0


# --------------------------------------------------------------------------- #
# Provenance string written to model.ec.source
# --------------------------------------------------------------------------- #

def test_source_records_fuzzy_wildcard_and_origin_detail():
    """A fuzzy BRENDA winner records its wildcard level and origin."""
    model = _ec_model(["r1", "r2"])
    df = pd.DataFrame({
        "rxn_id": ["r1", "r2"],
        "source": ["brenda", "brenda"],
        "kcat": [5.0, 7.0],
        "wildcard_level": pd.array([0, 2], dtype="Int64"),
        "origin": pd.array([1, 6], dtype="Int64"),
    })
    apply_kcat_list(model, df)
    assert model.ec.source[0] == "brenda (wc=0, origin=1)"
    assert model.ec.source[1] == "brenda (wc=2, origin=6)"


def test_source_winning_row_metadata_used_under_max():
    """The detail comes from the row whose kcat actually won."""
    model = _ec_model(["r1"])
    df = pd.DataFrame({
        "rxn_id": ["r1", "r1"],
        "source": ["brenda", "brenda"],
        "kcat": [5.0, 9.0],
        "wildcard_level": pd.array([2, 0], dtype="Int64"),
        "origin": pd.array([6, 1], dtype="Int64"),
    })
    apply_kcat_list(model, df, criteria="max")
    # 9.0 wins -> its (wc=0, origin=1) is recorded, not the 5.0 row's.
    assert model.ec.kcat[0] == 9.0
    assert model.ec.source[0] == "brenda (wc=0, origin=1)"


def test_source_bare_lowercase_token_without_metadata():
    """Exact DB hits and predictions (NA metadata) render as the bare,
    lowercased source token."""
    model = _ec_model(["r1", "r2"])
    df = pd.DataFrame({
        "rxn_id": ["r1", "r2"],
        "source": ["Sabio-RK", "CataPro"],
        "kcat": [5.0, 7.0],
        "wildcard_level": pd.array([pd.NA, pd.NA], dtype="Int64"),
        "origin": pd.array([pd.NA, pd.NA], dtype="Int64"),
    })
    apply_kcat_list(model, df)
    assert model.ec.source[0] == "sabio_rk"
    assert model.ec.source[1] == "catapro"


# --------------------------------------------------------------------------- #
# Criteria choice
# --------------------------------------------------------------------------- #

_THREE_CAND = [("r1", 1.0, "src_low"), ("r1", 4.0, "src_mid"),
               ("r1", 9.0, "src_high")]


def test_criteria_default_is_max():
    model = _ec_model(["r1"])
    apply_kcat_list(model, _kcat_list(_THREE_CAND))
    assert model.ec.kcat[0] == 9.0


def test_explicit_criteria_is_used():
    model = _ec_model(["r1"])
    apply_kcat_list(model, _kcat_list(_THREE_CAND), criteria="min")
    assert model.ec.kcat[0] == 1.0
