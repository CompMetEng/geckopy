"""Tests for fill_enz_concs."""
import numpy as np
import pytest
from scipy import sparse

from geckopy.databases import ProtData
from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import fill_enz_concs


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _ec_model(
    enzymes: list[str],
    *,
    initial_concs: list[float] | None = None,
) -> EcModel:
    g = len(enzymes)
    if initial_concs is None:
        initial_concs = [np.nan] * g
    model = EcModel("test")
    model.ec = EcData(
        rxns=[],
        kcat=np.empty(0, dtype=float),
        source=[],
        notes=[],
        eccodes=[],
        genes=list(enzymes),
        enzymes=list(enzymes),
        mw=np.zeros(g, dtype=float),
        sequence=[""] * g,
        concs=np.array(initial_concs, dtype=float),
        rxn_enz_mat=sparse.csr_matrix((0, g), dtype=float),
    )
    return model


def _prot_data(
    rows: list[tuple[str, float | list[float]]],
) -> ProtData:
    ids = [r[0] for r in rows]
    raw = [r[1] for r in rows]
    if all(isinstance(a, list) for a in raw):
        arr = np.array(raw, dtype=float)
    else:
        arr = np.array(raw, dtype=float)
    return ProtData(uniprot_ids=ids, abundances=arr)


# --------------------------------------------------------------------------- #
# 1D abundances
# --------------------------------------------------------------------------- #

def test_single_match_fills_one_slot():
    model = _ec_model(["P1", "P2"])
    pd = _prot_data([("P1", 5.0)])
    fill_enz_concs(model, pd)
    np.testing.assert_array_equal(model.ec.concs, [5.0, np.nan])


def test_all_match_fills_all_slots():
    model = _ec_model(["P1", "P2"])
    pd = _prot_data([("P1", 5.0), ("P2", 7.0)])
    fill_enz_concs(model, pd)
    np.testing.assert_array_equal(model.ec.concs, [5.0, 7.0])


def test_unmatched_uniprot_ids_ignored():
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 5.0), ("PX", 999.0)])
    fill_enz_concs(model, pd)
    np.testing.assert_array_equal(model.ec.concs, [5.0])


def test_enzymes_without_proteomic_data_stay_nan():
    model = _ec_model(["P1", "P2", "P3"])
    pd = _prot_data([("P2", 5.0)])
    fill_enz_concs(model, pd)
    assert np.isnan(model.ec.concs[0])
    assert model.ec.concs[1] == 5.0
    assert np.isnan(model.ec.concs[2])


def test_existing_concs_overwritten_with_nan_then_filled():
    """Pre-existing concs are reset to NaN first; then filled where
    proteomic data matches."""
    model = _ec_model(["P1", "P2"], initial_concs=[99.0, 88.0])
    pd = _prot_data([("P1", 5.0)])
    fill_enz_concs(model, pd)
    assert model.ec.concs[0] == 5.0
    # P2 had a value before but no proteomic match -> reset to NaN.
    assert np.isnan(model.ec.concs[1])


# --------------------------------------------------------------------------- #
# 2D abundances + data_col
# --------------------------------------------------------------------------- #

def test_2d_default_column_zero():
    model = _ec_model(["P1", "P2"])
    pd = ProtData(
        uniprot_ids=["P1", "P2"],
        abundances=np.array([[10.0, 100.0], [20.0, 200.0]]),
    )
    fill_enz_concs(model, pd)  # default data_col=0
    np.testing.assert_array_equal(model.ec.concs, [10.0, 20.0])


def test_2d_explicit_column():
    model = _ec_model(["P1", "P2"])
    pd = ProtData(
        uniprot_ids=["P1", "P2"],
        abundances=np.array([[10.0, 100.0], [20.0, 200.0]]),
    )
    fill_enz_concs(model, pd, data_col=1)
    np.testing.assert_array_equal(model.ec.concs, [100.0, 200.0])


def test_2d_out_of_range_data_col_raises():
    model = _ec_model(["P1"])
    pd = ProtData(
        uniprot_ids=["P1"],
        abundances=np.array([[10.0, 100.0]]),
    )
    with pytest.raises(IndexError, match="out of range"):
        fill_enz_concs(model, pd, data_col=5)


def test_1d_with_nonzero_data_col_raises():
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 5.0)])  # 1-D abundances
    with pytest.raises(IndexError, match="1-D"):
        fill_enz_concs(model, pd, data_col=1)


# --------------------------------------------------------------------------- #
# Trivial edge cases
# --------------------------------------------------------------------------- #

def test_empty_model_no_crash():
    model = _ec_model([])
    pd = _prot_data([("P1", 5.0)])
    fill_enz_concs(model, pd)
    assert model.ec.concs.size == 0


def test_empty_prot_data_resets_all_to_nan():
    model = _ec_model(["P1", "P2"], initial_concs=[99.0, 88.0])
    pd = _prot_data([])
    fill_enz_concs(model, pd)
    assert np.all(np.isnan(model.ec.concs))


def test_concs_length_matches_n_enzymes():
    model = _ec_model(["P1", "P2", "P3"])
    pd = _prot_data([("P1", 5.0)])
    fill_enz_concs(model, pd)
    assert len(model.ec.concs) == 3


# --------------------------------------------------------------------------- #
# Integration with calculate_f_factor / load_pax_db
# --------------------------------------------------------------------------- #

def test_integration_with_load_pax_db(tmp_path):
    """End-to-end: load a paxDB, fill concs, verify they line up with
    the ec.enzymes order."""
    from geckopy.databases import UniprotDB, load_pax_db

    pax = tmp_path / "paxDB.tsv"
    pax.write_text(
        "# hdr\n"
        "1\tg1\t2.0\n"  # P1 abundance = 2*10 = 20
        "2\tg2\t3.0\n"  # P2 abundance = 3*20 = 60
        "3\tg3\t1.0\n",  # P3 abundance = 1*30 = 30
        encoding="utf-8",
    )
    db = UniprotDB(
        ids=["P1", "P2", "P3"],
        genes=["g1", "g2", "g3"],
        eccodes=["", "", ""],
        mw=np.array([10.0, 20.0, 30.0]),
        sequences=["", "", ""],
    )
    prot = load_pax_db(pax, db)
    # Model has enzymes in a DIFFERENT order than the paxDB file.
    model = _ec_model(["P2", "P1"])
    fill_enz_concs(model, prot)
    # P2 -> index 0 (60), P1 -> index 1 (20).
    np.testing.assert_array_equal(model.ec.concs, [60.0, 20.0])


# --------------------------------------------------------------------------- #
# Negative abundance
# --------------------------------------------------------------------------- #

def test_negative_abundance_of_an_enzyme_raises():
    model = _ec_model(["P1", "P2"], initial_concs=[1.0, 2.0])
    pd = _prot_data([("P1", 5.0), ("P2", -0.5)])
    with pytest.raises(ValueError, match="negative abundance"):
        fill_enz_concs(model, pd)
    np.testing.assert_array_equal(model.ec.concs, [1.0, 2.0])


def test_negative_abundance_of_a_protein_outside_the_model_is_not_read():
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 5.0), ("X1", -1.0)])
    fill_enz_concs(model, pd)
    np.testing.assert_array_equal(model.ec.concs, [5.0])
