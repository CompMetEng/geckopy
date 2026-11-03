"""Tests for load_prot_data."""
from pathlib import Path

import numpy as np
import pytest

from geckopy.databases import ProtData, load_prot_data


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_empty_repl_per_cond_raises(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [["uniprot", "v1"], ["P1", "1.0"]])
    with pytest.raises(ValueError, match="repl_per_cond"):
        load_prot_data(p, [])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="proteomics file"):
        load_prot_data(tmp_path / "missing.tsv", [1])


def test_short_header_raises(tmp_path):
    p = tmp_path / "short.tsv"
    _write_tsv(p, [["uniprot"], ["P1"]])
    with pytest.raises(ValueError, match="header has"):
        load_prot_data(p, [1])


def test_header_count_mismatch_raises(tmp_path):
    """Header has 2 columns; repl_per_cond=[3] expects 4."""
    p = tmp_path / "wrong.tsv"
    _write_tsv(p, [["uniprot", "v1"], ["P1", "1.0"]])
    with pytest.raises(ValueError, match="repl_per_cond sums"):
        load_prot_data(p, [3])


# --------------------------------------------------------------------------- #
# Single condition, single replicate (matches the existing fixture format)
# --------------------------------------------------------------------------- #

def test_single_condition_single_replicate(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "level"],
        ["P1", "5.0"],
        ["P2", "10.0"],
    ])
    pd = load_prot_data(p, [1], filter_data=False)
    assert pd.uniprot_ids == ["P1", "P2"]
    assert pd.abundances.shape == (2, 1)
    np.testing.assert_array_equal(
        pd.abundances, [[5.0], [10.0]]
    )


def test_filter_data_false_just_takes_mean(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "v1", "v2", "v3"],
        ["P1", "10.0", "20.0", "30.0"],
    ])
    pd = load_prot_data(p, [3], filter_data=False)
    assert pd.abundances.shape == (1, 1)
    assert pd.abundances[0, 0] == pytest.approx(20.0)  # mean of 10/20/30


# --------------------------------------------------------------------------- #
# NaN parsing
# --------------------------------------------------------------------------- #

def test_nan_tokens_recognised(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "v1"],
        ["P1", "NA"],
        ["P2", "NaN"],
        ["P3", "#VALUE!"],
        ["P4", "5.0"],
    ])
    pd = load_prot_data(p, [1], filter_data=False)
    # P1, P2, P3 reduce to all-NaN -> dropped.
    assert pd.uniprot_ids == ["P4"]
    np.testing.assert_array_equal(pd.abundances, [[5.0]])


def test_empty_uniprot_dropped(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "v1"],
        ["", "5.0"],
        ["P1", "10.0"],
    ])
    pd = load_prot_data(p, [1], filter_data=False)
    assert pd.uniprot_ids == ["P1"]


# --------------------------------------------------------------------------- #
# Multi-condition
# --------------------------------------------------------------------------- #

def test_multiple_conditions_with_different_replicate_counts(tmp_path):
    """3 reps for cond 0, 2 reps for cond 1."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "c0_r1", "c0_r2", "c0_r3", "c1_r1", "c1_r2"],
        ["P1", "10.0", "20.0", "30.0", "100.0", "200.0"],
    ])
    pd = load_prot_data(p, [3, 2], filter_data=False)
    assert pd.abundances.shape == (1, 2)
    # Cond 0 mean = 20, cond 1 mean = 150.
    np.testing.assert_array_equal(pd.abundances, [[20.0, 150.0]])


# --------------------------------------------------------------------------- #
# Filter pipeline
# --------------------------------------------------------------------------- #

def test_max_missing_filter_drops_row_with_too_few_positive(tmp_path):
    """With 3 reps and max_missing=2/3, need >= 2 positive measurements."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2", "r3"],
        ["P_kept", "10", "20", "30"],
        ["P_dropped", "10", "0", "0"],  # only 1 positive
    ])
    pd = load_prot_data(
        p, [3], filter_data=True,
        max_missing=2/3, max_rsd=10.0, min_val=0.0,
        cut_lowest=0.0, add_stdevs=0.0,
    )
    # P_dropped collapsed to NaN, removed (all-NaN row).
    assert pd.uniprot_ids == ["P_kept"]


def test_max_missing_per_condition_list(tmp_path):
    """max_missing as a list with per-condition values."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "c0_r1", "c0_r2", "c1_r1", "c1_r2"],
        ["P1", "10", "0", "10", "0"],  # 1/2 positive each
    ])
    # Cond 0: max_missing=0.6 -> needs 1.2 -> 1 not enough -> dropped.
    # Cond 1: max_missing=0.4 -> needs 0.8 -> 1 enough -> kept.
    pd = load_prot_data(
        p, [2, 2], filter_data=True,
        max_missing=[0.6, 0.4], max_rsd=10.0, min_val=0.0,
        cut_lowest=0.0, add_stdevs=0.0,
    )
    assert pd.uniprot_ids == ["P1"]
    assert np.isnan(pd.abundances[0, 0])
    assert not np.isnan(pd.abundances[0, 1])


def test_max_rsd_filter_drops_high_variance_row(tmp_path):
    """RSD = std/mean. For [1, 100], std is ~70, mean is 50.5 -> RSD ~1.4."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2"],
        ["P_kept", "10", "11"],          # low RSD
        ["P_dropped", "1", "100"],       # high RSD
    ])
    pd = load_prot_data(
        p, [2], filter_data=True,
        max_missing=0.0, max_rsd=0.5, min_val=0.0,
        cut_lowest=0.0, add_stdevs=0.0,
    )
    assert pd.uniprot_ids == ["P_kept"]


def test_min_val_filter_drops_low_collapsed_value(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1"],
        ["P_low", "1.0"],
        ["P_high", "100.0"],
    ])
    pd = load_prot_data(
        p, [1], filter_data=True,
        max_missing=0.0, max_rsd=10.0, min_val=10.0,
        cut_lowest=0.0, add_stdevs=0.0,
    )
    assert pd.uniprot_ids == ["P_high"]


def test_cut_lowest_drops_bottom_percent(tmp_path):
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1"],
        *[[f"P{i}", str(float(i))] for i in range(1, 21)],  # P1..P20 = 1..20
    ])
    pd = load_prot_data(
        p, [1], filter_data=True,
        max_missing=0.0, max_rsd=10.0, min_val=0.0,
        cut_lowest=20.0, add_stdevs=0.0,  # cut bottom 20% = 4 lowest
    )
    # Bottom 4 (P1-P4) dropped. P5-P20 kept.
    assert set(pd.uniprot_ids) == {f"P{i}" for i in range(5, 21)}


def test_add_stdevs_inflates_collapsed_value(tmp_path):
    """Collapsed value = mean + add_stdevs * std. For [10, 20, 30],
    mean=20, std (ddof=1)=10. With add_stdevs=2, collapsed=40."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2", "r3"],
        ["P1", "10", "20", "30"],
    ])
    pd = load_prot_data(
        p, [3], filter_data=True,
        max_missing=0.0, max_rsd=10.0, min_val=0.0,
        cut_lowest=0.0, add_stdevs=2.0,
    )
    assert pd.abundances[0, 0] == pytest.approx(40.0)


# --------------------------------------------------------------------------- #
# Re-filtering: pass a ProtData
# --------------------------------------------------------------------------- #

def test_source_can_be_existing_prot_data(tmp_path):
    """Re-filter an already-loaded ProtData (matches MATLAB)."""
    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2", "r3"],
        ["P1", "10", "20", "30"],
        ["P2", "1", "2", "3"],
    ])
    initial = load_prot_data(p, [3], filter_data=False)
    # Wrap as 2D source (re-filter requires 2D).
    pd2 = load_prot_data(initial, [1], filter_data=False)
    assert pd2.uniprot_ids == ["P1", "P2"]


# --------------------------------------------------------------------------- #
# Integration with calculate_f_factor
# --------------------------------------------------------------------------- #

def test_loaded_data_can_feed_calculate_f_factor(tmp_path):
    from geckopy import EcModel
    from geckopy.ec_model.ec_data import EcData
    from geckopy.limit_proteins import calculate_f_factor
    from scipy import sparse

    p = tmp_path / "p.tsv"
    _write_tsv(p, [
        ["uniprot", "r1"],
        ["P1", "10.0"],
        ["P2", "5.0"],
    ])
    pd = load_prot_data(p, [1], filter_data=False)

    model = EcModel("test")
    model.ec = EcData(
        rxns=[], kcat=np.empty(0), source=[], notes=[], eccodes=[],
        genes=[], enzymes=["P1"], mw=np.array([100.0]),
        sequence=[""], concs=np.array([np.nan]),
        rxn_enz_mat=sparse.csr_matrix((0, 1)),
    )
    f = calculate_f_factor(model, pd)
    # P1 in model contributes 10; total = 15. f = 10/15 ~= 0.667.
    assert f == pytest.approx(10.0 / 15.0)


# --------------------------------------------------------------------------- #
# zeros_as_measured
# --------------------------------------------------------------------------- #

_ZERO_ROWS = [
    ["uniprot", "r1", "r2", "r3"],
    ["Z1", "0", "0", "0"],
    ["Z2", "0", "4", "6"],
    ["Z3", "5", "5", "5"],
]


@pytest.mark.parametrize("filter_data", [True, False])
def test_zero_is_not_measured_by_default(tmp_path, filter_data):
    """Z1 is 0 in every replicate; Z2 is 0 in one replicate. Both are read as
    not measured: Z1 is dropped, and the mean of Z2 is over the two values."""
    p = tmp_path / "z.tsv"
    _write_tsv(p, _ZERO_ROWS)
    pd = load_prot_data(
        p, [3], filter_data=filter_data, add_stdevs=0.0, cut_lowest=0.0,
    )
    assert pd.uniprot_ids == ["Z2", "Z3"]
    np.testing.assert_allclose(pd.abundances, [[5.0], [5.0]], atol=1e-12)


@pytest.mark.parametrize("filter_data", [True, False])
def test_zeros_as_measured_keeps_zero(tmp_path, filter_data):
    """With zeros_as_measured Z1 stays 0 and the 0 of Z2 enters its mean."""
    p = tmp_path / "z.tsv"
    _write_tsv(p, _ZERO_ROWS)
    pd = load_prot_data(
        p, [3], filter_data=filter_data, add_stdevs=0.0, cut_lowest=0.0,
        zeros_as_measured=True,
    )
    assert pd.uniprot_ids == ["Z1", "Z2", "Z3"]
    np.testing.assert_allclose(pd.abundances, [[0.0], [10.0 / 3.0], [5.0]], atol=1e-12)


def test_measured_zero_is_exempt_from_min_val_and_cut_lowest(tmp_path):
    """P1 is 0; P2 to P10 are 2 to 10. min_val 2.5 removes P2, then
    cut_lowest 20 percent removes one of the eight positive values (P3) and
    not P1."""
    p = tmp_path / "z.tsv"
    rows = [["uniprot", "r1", "r2", "r3"], ["P1", "0", "0", "0"]]
    rows += [[f"P{k}", str(float(k)), str(float(k)), str(float(k))] for k in range(2, 11)]
    _write_tsv(p, rows)
    pd = load_prot_data(
        p, [3], add_stdevs=0.0, min_val=2.5, cut_lowest=20.0, zeros_as_measured=True,
    )
    assert pd.uniprot_ids == ["P1"] + [f"P{k}" for k in range(4, 11)]
    np.testing.assert_allclose(
        pd.abundances[:, 0], [0.0] + [float(k) for k in range(4, 11)], atol=1e-12,
    )
    default = load_prot_data(
        p, [3], add_stdevs=0.0, min_val=2.5, cut_lowest=20.0,
    )
    assert default.uniprot_ids == [f"P{k}" for k in range(4, 11)]


def test_prot_data_source_reads_zero_as_not_measured_by_default():
    source = ProtData(
        uniprot_ids=["Z1", "Z2", "Z3"],
        abundances=np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 6.0], [5.0, 5.0, 5.0]]),
    )
    original = source.abundances.copy()
    pd = load_prot_data(source, [3], add_stdevs=0.0, cut_lowest=0.0)
    assert pd.uniprot_ids == ["Z2", "Z3"]
    np.testing.assert_allclose(pd.abundances, [[5.0], [5.0]], atol=1e-12)
    pd = load_prot_data(source, [3], add_stdevs=0.0, cut_lowest=0.0, zeros_as_measured=True)
    assert pd.uniprot_ids == ["Z1", "Z2", "Z3"]
    np.testing.assert_allclose(pd.abundances, [[0.0], [10.0 / 3.0], [5.0]], atol=1e-12)
    np.testing.assert_array_equal(source.abundances, original)


def test_max_missing_counts_a_zero_replicate_only_with_zeros_as_measured(tmp_path):
    """A = [0, 5, 5] has two positive replicates of three, so max_missing 1.0
    removes it by default; with zeros_as_measured all three are present, the
    RSD is 2.887 / 3.333 = 0.87 and the mean is 10/3."""
    p = tmp_path / "m.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2", "r3"],
        ["A", "0", "5", "5"],
        ["B", "5", "5", "5"],
    ])
    kwargs = dict(max_missing=1.0, add_stdevs=0.0, cut_lowest=0.0)
    pd = load_prot_data(p, [3], **kwargs)
    assert pd.uniprot_ids == ["B"]
    pd = load_prot_data(p, [3], zeros_as_measured=True, **kwargs)
    assert pd.uniprot_ids == ["A", "B"]
    np.testing.assert_allclose(pd.abundances, [[10.0 / 3.0], [5.0]], atol=1e-12)


def test_zero_gives_nan_in_one_condition_and_a_value_in_the_other(tmp_path):
    p = tmp_path / "c.tsv"
    _write_tsv(p, [
        ["uniprot", "r1", "r2", "r3", "r4"],
        ["P", "0", "0", "3", "5"],
        ["Q", "2", "2", "2", "2"],
    ])
    kwargs = dict(add_stdevs=0.0, cut_lowest=0.0)
    pd = load_prot_data(p, [2, 2], **kwargs)
    assert pd.uniprot_ids == ["P", "Q"]
    np.testing.assert_allclose(pd.abundances, [[np.nan, 4.0], [2.0, 2.0]], atol=1e-12)
    pd = load_prot_data(p, [2, 2], zeros_as_measured=True, **kwargs)
    np.testing.assert_allclose(pd.abundances, [[0.0, 4.0], [2.0, 2.0]], atol=1e-12)


@pytest.mark.parametrize("zeros_as_measured", [False, True])
def test_cut_lowest_removes_the_first_of_tied_values(tmp_path, zeros_as_measured):
    """Ten values 1,1,2,2,...,5,5; cut_lowest 10 percent removes one value,
    the first of the two lowest (P1)."""
    p = tmp_path / "t.tsv"
    _write_tsv(p, [
        ["uniprot", "r1"],
        *[[f"P{i}", str(float(v))] for i, v in enumerate([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 1)],
    ])
    pd = load_prot_data(
        p, [1], add_stdevs=0.0, cut_lowest=10.0, zeros_as_measured=zeros_as_measured,
    )
    assert pd.uniprot_ids == [f"P{i}" for i in range(2, 11)]
