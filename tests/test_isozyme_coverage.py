"""Tests for get_isozyme_coverage, zero_unmeasured_isozymes and the measured-zero handling of flexibilize_enz_concs."""
import logging

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import (
    constrain_enz_concs,
    flexibilize_enz_concs,
    get_isozyme_coverage,
    zero_unmeasured_isozymes,
)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def _isozyme_model(concs) -> EcModel:
    """Five enzymes with usage reactions (upper bound 1000) and the entries
    X, X_EXP_1: P1, P2;  Y, Y_EXP_1: P3, P4;  Z: P4;  W, W_EXP_1: P5, P1."""
    enzymes = ["P1", "P2", "P3", "P4", "P5"]
    model = EcModel("isozymes")
    reactions = []
    for enzyme in enzymes:
        rxn = cobra.Reaction(f"usage_prot_{enzyme}")
        rxn.bounds = (0.0, 1000.0)
        reactions.append(rxn)
    model.add_reactions(reactions)
    model.ec = EcData(
        rxns=["X", "X_EXP_1", "Y", "Y_EXP_1", "Z", "W", "W_EXP_1"],
        enzymes=enzymes,
        concs=np.array(concs, dtype=float),
        rxn_enz_mat=sparse.csr_matrix(np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ], dtype=float)),
    )
    return model


def _usage_ub(model, enzymes):
    return [model.reactions.get_by_id(f"usage_prot_{e}").upper_bound for e in enzymes]


def _zero_toy(concs) -> EcModel:
    """A -> B by E1 or by E2 (R_AB1, R_AB2); growth (R4) equals the flux."""
    model = EcModel("toy")
    mets = {n: cobra.Metabolite(n, compartment="c") for n in
            ["A_e", "A_c", "B_c", "prot_pool", "prot_E1", "prot_E2"]}
    model.add_metabolites(list(mets.values()))
    spec = [
        ("EX_A", {"A_e": -1}, -1000, 0),
        ("TR_A", {"A_e": -1, "A_c": 1}, 0, 1000),
        ("R_AB1", {"A_c": -1, "B_c": 1, "prot_E1": -1}, 0, 1000),
        ("R_AB2", {"A_c": -1, "B_c": 1, "prot_E2": -1}, 0, 1000),
        ("R4", {"B_c": -1}, 0, 1000),
        ("prot_pool_exchange", {"prot_pool": 1}, 0, 1000),
        ("usage_prot_E1", {"prot_pool": -1, "prot_E1": 1}, 0, 1000),
        ("usage_prot_E2", {"prot_pool": -1, "prot_E2": 1}, 0, 1000),
    ]
    for rxn_id, stoich, lb, ub in spec:
        rxn = cobra.Reaction(rxn_id)
        rxn.add_metabolites({mets[k]: float(v) for k, v in stoich.items()})
        rxn.bounds = (float(lb), float(ub))
        model.add_reactions([rxn])
    model.objective = "R4"
    model.ec = EcData(
        rxns=["R_AB1", "R_AB2"],
        kcat=np.array([1.0, 1.0]),
        source=["", ""],
        notes=["", ""],
        eccodes=["", ""],
        genes=["g_E1", "g_E2"],
        enzymes=["E1", "E2"],
        mw=np.array([100.0, 100.0]),
        sequence=["", ""],
        concs=np.array(concs, dtype=float),
        rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    constrain_enz_concs(model)
    return model


# --------------------------------------------------------------------------- #
# get_isozyme_coverage
# --------------------------------------------------------------------------- #

def test_coverage_reports_partial_groups():
    """W, X and Y have a measured and an unmeasured entry; Z has one entry
    only. P4 also catalyses Z, so zeroing it would limit Z as well."""
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    coverage, zeroable = get_isozyme_coverage(model)
    assert list(coverage.columns) == [
        "rxn", "n_entries", "n_measured", "n_unmeasured",
        "measured_enzymes", "unmeasured_enzymes", "shared_enzymes",
    ]
    assert list(coverage["rxn"]) == ["W", "X", "Y"]
    assert list(coverage["n_entries"]) == [2, 2, 2]
    assert list(coverage["n_measured"]) == [1, 1, 1]
    assert list(coverage["n_unmeasured"]) == [1, 1, 1]
    assert list(coverage["measured_enzymes"]) == [["P1"], ["P1"], ["P3"]]
    assert list(coverage["unmeasured_enzymes"]) == [["P5"], ["P2"], ["P4"]]
    assert list(coverage["shared_enzymes"]) == [[], [], ["P4"]]
    assert zeroable == ["P2", "P5"]


def test_coverage_counts_a_measured_subunit_as_measured_entry():
    """The entry V has the subunits P1 (measured) and P2 (not measured): it is
    a measured entry, and P2 is not an unmeasured isozyme."""
    model = _isozyme_model([3, np.nan, np.nan, np.nan, np.nan])
    model.ec.rxns = ["V", "V_EXP_1"]
    model.ec.rxn_enz_mat = sparse.csr_matrix(np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ], dtype=float))
    coverage, zeroable = get_isozyme_coverage(model)
    assert list(coverage["rxn"]) == ["V"]
    assert list(coverage["measured_enzymes"]) == [["P1"]]
    assert list(coverage["unmeasured_enzymes"]) == [["P4", "P5"]]
    assert zeroable == ["P4", "P5"]


def test_coverage_ignores_measured_zero():
    """A concentration of 0 is neither a measured sibling nor an unmeasured entry."""
    model = _isozyme_model([0, np.nan, 2, np.nan, np.nan])
    coverage, zeroable = get_isozyme_coverage(model)
    assert list(coverage["rxn"]) == ["Y"]
    assert zeroable == []


def test_coverage_empty_without_measurements():
    model = _isozyme_model([np.nan] * 5)
    coverage, zeroable = get_isozyme_coverage(model)
    assert len(coverage) == 0
    assert zeroable == []
    model.ec.concs = np.empty(0)
    assert len(get_isozyme_coverage(model)[0]) == 0


def test_coverage_short_concs_count_as_nan():
    model = _isozyme_model([3])
    coverage, zeroable = get_isozyme_coverage(model)
    full, full_zeroable = get_isozyme_coverage(_isozyme_model([3] + [np.nan] * 4))
    assert coverage.equals(full)
    assert zeroable == full_zeroable


def test_coverage_keeps_enzyme_of_measured_entry_shared():
    """P2 is not measured, and it is in the measured entry X (with P1) as well
    as in the unmeasured entry X_EXP_1, so it is shared and not zeroable."""
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    dense = model.ec.rxn_enz_mat.toarray()
    dense[0, :] = [1, 1, 0, 0, 0]
    model.ec.rxn_enz_mat = sparse.csr_matrix(dense)
    coverage, zeroable = get_isozyme_coverage(model)
    assert coverage.loc[coverage["rxn"] == "X", "shared_enzymes"].iloc[0] == ["P2"]
    assert zeroable == ["P5"]


def test_coverage_without_concs_is_empty():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    del model.ec.concs
    coverage, zeroable = get_isozyme_coverage(model)
    assert len(coverage) == 0
    assert zeroable == []


def test_coverage_does_not_modify_model():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    concs = model.ec.concs.copy()
    get_isozyme_coverage(model)
    assert np.array_equal(model.ec.concs, concs, equal_nan=True)
    assert _usage_ub(model, ["P1", "P2", "P3", "P4", "P5"]) == [1000.0] * 5


def test_coverage_rejects_light_model():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    model.ec.gecko_light = True
    with pytest.raises(NotImplementedError, match="gecko-light"):
        get_isozyme_coverage(model)


def test_coverage_rejects_missing_rxn_enz_mat():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    model.ec.rxn_enz_mat = sparse.csr_matrix((0, 0))
    with pytest.raises(ValueError, match="rxn_enz_mat"):
        get_isozyme_coverage(model)


# --------------------------------------------------------------------------- #
# zero_unmeasured_isozymes
# --------------------------------------------------------------------------- #

def test_zero_sets_usage_bound_to_zero():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    concs = model.ec.concs.copy()
    changed = zero_unmeasured_isozymes(model)
    assert list(changed.columns) == ["uniprot", "new_upper_bound"]
    assert list(changed["uniprot"]) == ["P2", "P5"]
    assert list(changed["new_upper_bound"]) == [0.0, 0.0]
    assert _usage_ub(model, ["P1", "P2", "P3", "P4", "P5"]) == [1000, 0, 1000, 1000, 0]
    assert np.array_equal(model.ec.concs, concs, equal_nan=True)


def test_detection_limit():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    # The default limit is the smallest concentration above 0
    default = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    zero_unmeasured_isozymes(default, method="detection_limit")
    assert _usage_ub(default, ["P2", "P5"]) == [2, 2]
    limited = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    zero_unmeasured_isozymes(limited, method="detection_limit", limit=0.5)
    assert _usage_ub(limited, ["P2", "P5", "P4"]) == [0.5, 0.5, 1000]
    # An upper bound below the limit is not raised
    model.reactions.get_by_id("usage_prot_P2").upper_bound = 0.1
    zero_unmeasured_isozymes(model, method="detection_limit", limit=0.5)
    assert _usage_ub(model, ["P2", "P5"]) == [0.1, 0.5]


def test_zero_leaves_shared_enzyme_unchanged():
    model = _isozyme_model([0, np.nan, 2, np.nan, np.nan])
    changed = zero_unmeasured_isozymes(model)
    # Y is partial but P4 is shared with Z, so nothing is zeroable
    assert len(changed) == 0
    assert _usage_ub(model, ["P1", "P2", "P3", "P4", "P5"]) == [1000.0] * 5


def test_zero_rejects_unknown_method_and_bad_limit():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    with pytest.raises(ValueError, match="method"):
        zero_unmeasured_isozymes(model, method="half")
    with pytest.raises(ValueError, match="detection limit"):
        zero_unmeasured_isozymes(model, method="detection_limit", limit=-1)
    with pytest.raises(ValueError, match="detection limit"):
        zero_unmeasured_isozymes(model, method="detection_limit", limit=float("nan"))
    no_concs = _isozyme_model([np.nan] * 5)
    with pytest.raises(ValueError, match="detection limit"):
        zero_unmeasured_isozymes(no_concs, method="detection_limit")


def test_zero_rejects_light_model():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    model.ec.gecko_light = True
    with pytest.raises(NotImplementedError, match="gecko-light"):
        zero_unmeasured_isozymes(model)


def test_zero_requires_usage_reactions():
    model = _isozyme_model([3, np.nan, 2, np.nan, np.nan])
    # P2 and P5 are zeroable; the reaction of the last one is missing
    model.remove_reactions(["usage_prot_P5"])
    with pytest.raises(ValueError, match="usage reaction"):
        zero_unmeasured_isozymes(model)
    assert _usage_ub(model, ["P1", "P2", "P3", "P4"]) == [1000.0] * 4


# --------------------------------------------------------------------------- #
# flexibilize_enz_concs with a measured zero
# --------------------------------------------------------------------------- #

def test_flexibilize_leaves_measured_zero_at_zero(caplog):
    """E1 is measured at 1 and limits growth; E2 is measured at 0 on a
    parallel route. E1 is raised, E2 stays at 0 and the function warns."""
    # An enzyme at 0 is excluded from the result by the mask in `_build_result` (see test_level_complex_concs.py for the per-complex case).
    model = _zero_toy([1, 0])
    with caplog.at_level(logging.WARNING):
        result = flexibilize_enz_concs(model, exp_growth=2.0, bio_rxn="R4", verbose=False)
    assert "enzymes E2 have a concentration of 0" in caplog.text
    assert result.uniprot_ids == ["E1"]
    assert _usage_ub(model, ["E2"]) == [0.0]
    assert model.optimize().objective_value == pytest.approx(2.0, rel=5e-3)


def test_flexibilize_cannot_raise_measured_zero():
    """The only route is measured at 0: no enzyme is limiting, the bound stays 0."""
    # An enzyme at 0 is excluded from the result by the mask in `_build_result` (see test_level_complex_concs.py for the per-complex case).
    model = _zero_toy([np.nan, 0])
    model.reactions.get_by_id("R_AB1").upper_bound = 0.0
    result = flexibilize_enz_concs(model, exp_growth=2.0, bio_rxn="R4", verbose=False)
    assert result.uniprot_ids == []
    assert _usage_ub(model, ["E2"]) == [0.0]
    assert model.optimize().objective_value == pytest.approx(0.0, abs=1e-9)


def test_flexibilize_warning_lists_at_most_ten_enzymes(caplog):
    model = _zero_toy([1, 0])
    names = [f"Q{i}" for i in range(12)]
    model.ec.enzymes = ["E1"] + names
    model.ec.concs = np.array([1.0] + [0.0] * 12)
    model.ec.mw = np.ones(13)
    with caplog.at_level(logging.WARNING):
        flexibilize_enz_concs(model, exp_growth=0.5, bio_rxn="R4", verbose=False)
    assert "Q9" in caplog.text and "Q10" not in caplog.text
    assert "and 2 more" in caplog.text


def test_flexibilize_does_not_warn_without_measured_zero(caplog):
    model = _zero_toy([1, 1])
    with caplog.at_level(logging.WARNING):
        flexibilize_enz_concs(model, exp_growth=2.0, bio_rxn="R4", verbose=False)
    assert "concentration of 0" not in caplog.text
