"""Tests for calculate_f_factor."""
import numpy as np
import pytest
from scipy import sparse

from geckopy.databases import ProtData
from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import calculate_f_factor


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _ec_model(enzymes: list[str]) -> EcModel:
    """Build a minimal EcModel with the given ec.enzymes."""
    g = len(enzymes)
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
        concs=np.full(g, np.nan, dtype=float),
        rxn_enz_mat=sparse.csr_matrix((0, g), dtype=float),
    )
    return model


def _prot_data(
    rows: list[tuple[str, float | list[float]]],
) -> ProtData:
    """Build ProtData from (uniprot_id, abundance) tuples.
    abundance can be a float (1D) or a list (2D)."""
    ids = [r[0] for r in rows]
    raw = [r[1] for r in rows]
    if all(isinstance(a, list) for a in raw):
        arr = np.array(raw, dtype=float)
    else:
        arr = np.array(raw, dtype=float)
    return ProtData(uniprot_ids=ids, abundances=arr)


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_all_enzymes_in_model_yields_one():
    model = _ec_model(["P1", "P2"])
    pd = _prot_data([("P1", 100.0), ("P2", 200.0)])
    assert calculate_f_factor(model, pd) == pytest.approx(1.0)


def test_no_enzymes_in_model_yields_zero():
    model = _ec_model([])
    pd = _prot_data([("P1", 100.0), ("P2", 200.0)])
    assert calculate_f_factor(model, pd) == 0.0


def test_subset_match_yields_correct_ratio():
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 100.0), ("P2", 300.0)])
    # P1 contributes 100 of 400 total -> 0.25.
    assert calculate_f_factor(model, pd) == pytest.approx(0.25)


def test_empty_proteome_yields_nan():
    model = _ec_model(["P1"])
    pd = _prot_data([])
    assert np.isnan(calculate_f_factor(model, pd))


def test_zero_total_proteome_yields_nan():
    """All abundances are zero -> the factor is 0/0, NaN as in MATLAB."""
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 0.0), ("P2", 0.0)])
    assert np.isnan(calculate_f_factor(model, pd))


# --------------------------------------------------------------------------- #
# Default enzymes from model.ec.enzymes
# --------------------------------------------------------------------------- #

def test_default_enzymes_uses_model_ec_enzymes():
    model = _ec_model(["P1", "P3"])
    pd = _prot_data([
        ("P1", 100.0),
        ("P2", 200.0),
        ("P3", 300.0),
    ])
    # P1 + P3 = 400, total = 600 -> f ~ 0.667
    assert calculate_f_factor(model, pd) == pytest.approx(400.0 / 600.0)


def test_explicit_enzymes_overrides_default():
    model = _ec_model(["P1", "P3"])
    pd = _prot_data([
        ("P1", 100.0),
        ("P2", 200.0),
        ("P3", 300.0),
    ])
    # Override: only P2 counts.
    f = calculate_f_factor(model, pd, enzymes=["P2"])
    assert f == pytest.approx(200.0 / 600.0)


# --------------------------------------------------------------------------- #
# 2D abundances (multiple samples)
# --------------------------------------------------------------------------- #

def test_2d_abundances_averaged_per_protein():
    model = _ec_model(["P1"])
    # Two samples per protein: P1 = (100, 200), P2 = (300, 100).
    pd = ProtData(
        uniprot_ids=["P1", "P2"],
        abundances=np.array([[100.0, 200.0], [300.0, 100.0]]),
    )
    # Avg: P1 = 150, P2 = 200. Total = 350. P1 fraction = 150/350.
    assert calculate_f_factor(model, pd) == pytest.approx(150.0 / 350.0)


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #

def test_nan_abundances_skipped_in_total():
    model = _ec_model(["P1"])
    pd = _prot_data([("P1", 100.0), ("P2", float("nan")), ("P3", 200.0)])
    # NaN is skipped -> total = 300, P1 = 100, f = 1/3.
    assert calculate_f_factor(model, pd) == pytest.approx(100.0 / 300.0)


def test_2d_nan_handled_per_sample():
    model = _ec_model(["P1"])
    pd = ProtData(
        uniprot_ids=["P1", "P2"],
        abundances=np.array([
            [100.0, np.nan],   # P1: nanmean = 100
            [np.nan, 400.0],   # P2: nanmean = 400
        ]),
    )
    # P1=100, P2=400. f = 100/500 = 0.2.
    assert calculate_f_factor(model, pd) == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Integration with load_pax_db
# --------------------------------------------------------------------------- #

def test_integration_with_pax_db_loader(tmp_path):
    """End-to-end: write a paxDB.tsv, load via load_pax_db, compute f."""
    from geckopy.databases import UniprotDB, load_pax_db

    pax_path = tmp_path / "paxDB.tsv"
    pax_path.write_text(
        "# header line\n"
        "1\tg1\t2.0\n"
        "2\tg2\t3.0\n"
        "3\tg3\t1.0\n",
        encoding="utf-8",
    )
    db = UniprotDB(
        ids=["P1", "P2", "P3"],
        genes=["g1", "g2", "g3"],
        eccodes=["", "", ""],
        mw=np.array([10.0, 20.0, 30.0]),
        sequences=["", "", ""],
    )
    pd_data = load_pax_db(pax_path, db)
    # Abundances: P1=2*10=20, P2=3*20=60, P3=1*30=30. Total=110.
    model = _ec_model(["P1", "P2"])
    f = calculate_f_factor(model, pd_data)
    # P1+P2 = 80, total = 110.
    assert f == pytest.approx(80.0 / 110.0)


# --------------------------------------------------------------------------- #
# Explicit empty enzyme list
# --------------------------------------------------------------------------- #

def test_explicit_empty_enzyme_list_means_no_enzymes():
    model = _ec_model(["P1", "P2"])
    pd = _prot_data([("P1", 100.0), ("P2", 100.0)])
    assert calculate_f_factor(model, pd, enzymes=[]) == 0.0
    assert calculate_f_factor(model, pd) == pytest.approx(1.0)
