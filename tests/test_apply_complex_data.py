"""Tests for apply_complex_data."""
import logging
from pathlib import Path

import cobra
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.databases import ComplexPortalEntry
from geckopy.ec_model.pipeline import (
    apply_complex_data,
    set_kcat_for_reactions,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


_ECTESTGEM_CACHE: EcModel | None = None


def _ectestgem_ec_model() -> EcModel:
    """Cached build of the ecTestGEM ecModel; deep-copied per call."""
    import copy as _copy
    global _ECTESTGEM_CACHE
    if _ECTESTGEM_CACHE is None:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _ECTESTGEM_CACHE = make_ec_model(cobra_model, adapter)
    return _copy.deepcopy(_ECTESTGEM_CACHE)


def _stoich(model: EcModel, rxn_id: str, enzyme: str) -> float:
    """Return rxn_enz_mat[rxn, enzyme] as float."""
    ri = model.ec.rxns.index(rxn_id)
    ej = model.ec.enzymes.index(enzyme)
    return float(model.ec.rxn_enz_mat[ri, ej])


# --------------------------------------------------------------------------- #
# Default behavior on the ecTestGEM fixture
# --------------------------------------------------------------------------- #

def test_default_path_loads_from_adapter_data_folder():
    """Without an explicit path, the function reads ComplexPortal.json
    from the adapter's data folder."""
    ec_model = _ectestgem_ec_model()
    # Confirm the prior state: rxn_enz_mat[R2_EXP_1, P1/P2] == 1.
    assert _stoich(ec_model, "R2_EXP_1", "P1") == 1.0
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 1.0

    apply_complex_data(ec_model, apply=False)

    # After applying, R2_EXP_1 should have stoichiometry [1, 2] from the
    # R2Compl complex.
    assert _stoich(ec_model, "R2_EXP_1", "P1") == 1.0
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 2.0


def test_applies_to_both_forward_and_reverse_isozymes():
    """The R2Compl complex should match R2_EXP_1 (forward) and
    R2_REV_EXP_1 (reverse), both with GPR 'G1 and G2'."""
    ec_model = _ectestgem_ec_model()
    apply_complex_data(ec_model, apply=False)

    assert _stoich(ec_model, "R2_REV_EXP_1", "P1") == 1.0
    assert _stoich(ec_model, "R2_REV_EXP_1", "P2") == 2.0


def test_does_not_touch_reactions_with_no_complex_match():
    """R3 has GPR G4 only, R5 has G5 only. Neither matches any complex."""
    ec_model = _ectestgem_ec_model()
    pre_r3 = _stoich(ec_model, "R3", "P4")
    pre_r5 = _stoich(ec_model, "R5", "P5")

    apply_complex_data(ec_model, apply=False)

    assert _stoich(ec_model, "R3", "P4") == pre_r3
    assert _stoich(ec_model, "R5", "P5") == pre_r5


def test_does_not_touch_single_isozyme_branches():
    """R2_EXP_2 has GPR G3 (single enzyme P3); no complex match. Default 1."""
    ec_model = _ectestgem_ec_model()
    apply_complex_data(ec_model, apply=False)
    assert _stoich(ec_model, "R2_EXP_2", "P3") == 1.0


# --------------------------------------------------------------------------- #
# Pre-loaded complex_data argument
# --------------------------------------------------------------------------- #

def test_accepts_preloaded_complex_data():
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="R2Compl",
        name="R2 complex",
        species="testus",
        gene_names=["G1", "G2"],
        protein_ids=["P1", "P2"],
        stoichiometry=[3, 4],
    )]
    apply_complex_data(ec_model, complex_data=data, apply=False)

    assert _stoich(ec_model, "R2_EXP_1", "P1") == 3.0
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 4.0


def test_empty_complex_data_is_noop(caplog):
    ec_model = _ectestgem_ec_model()
    with caplog.at_level(logging.INFO):
        apply_complex_data(ec_model, complex_data=[], apply=False)
    assert "nothing to do" in caplog.text


# --------------------------------------------------------------------------- #
# Proposed (non-applied) cases
# --------------------------------------------------------------------------- #

def test_superset_complex_logs_proposal(caplog):
    """Model has {P1, P2}; complex has {P1, P2, P_other}. Model is a
    strict subset; should be reported as a superset proposal, not applied."""
    ec_model = _ectestgem_ec_model()

    # Build a complex that includes an extra protein P_extra not in the model.
    data = [ComplexPortalEntry(
        complex_id="C_super",
        name="superset",
        species="t",
        gene_names=["G1", "G2", "G_extra"],
        protein_ids=["P1", "P2", "P_extra"],
        stoichiometry=[1, 1, 1],
    )]

    with caplog.at_level(logging.WARNING):
        apply_complex_data(ec_model, complex_data=data, apply=False)

    assert "Proposed complex" in caplog.text
    assert "C_super" in caplog.text
    # Should not have been applied.
    assert _stoich(ec_model, "R2_EXP_1", "P1") == 1.0  # default, unchanged


def test_partial_match_logs_proposal(caplog):
    """Model has {P1, P2}; complex has {P1, P_other}. 50% match: below
    the default threshold of 0.75 so should NOT log."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="C_partial",
        name="partial",
        species="t",
        gene_names=["G1"],
        protein_ids=["P1"],
        stoichiometry=[1],
    )]

    with caplog.at_level(logging.WARNING):
        apply_complex_data(ec_model, complex_data=data, apply=False)

    # 50% match for R2_EXP_1 (P1 in {P1,P2}) is below 0.75 threshold.
    assert "C_partial" not in caplog.text


def test_partial_match_above_threshold_logs(caplog):
    """Lower the threshold to capture the 50% case."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="C_partial",
        name="partial",
        species="t",
        gene_names=["G1"],
        protein_ids=["P1"],
        stoichiometry=[1],
    )]
    with caplog.at_level(logging.WARNING):
        apply_complex_data(
            ec_model, complex_data=data, apply=False,
            min_match_to_propose=0.5,
        )
    assert "C_partial" in caplog.text


def test_single_enzyme_reaction_not_proposed(caplog):
    """Single-enzyme reactions are not eligible for proposed complexes,
    since their stoichiometry is already unambiguous."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="C_extra",
        name="extra",
        species="t",
        gene_names=["G4", "G_other"],
        protein_ids=["P4", "P_other"],
        stoichiometry=[1, 1],
    )]
    with caplog.at_level(logging.WARNING):
        apply_complex_data(ec_model, complex_data=data, apply=False)
    # R3 has only P4; should not be proposed even though P4 is in C_extra.
    assert "C_extra" not in caplog.text


# --------------------------------------------------------------------------- #
# Apply parameter and S matrix
# --------------------------------------------------------------------------- #

def test_apply_true_recomputes_kcat_coefficients():
    """Setting a kcat first, then applying complex data with apply=True,
    should give a coefficient that uses the new stoichiometry."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=True)

    # Pre-state: P2 stoich is 1, so coefficient = -(1 * 20000 / (100 * 3600)).
    r = ec_model.reactions.get_by_id("R2_EXP_1")
    pre_coef = next(c for m, c in r.metabolites.items() if m.id == "prot_P2")
    expected_pre = -(1 * 20000.0 / (100.0 * 3600.0))
    assert pre_coef == pytest.approx(expected_pre)

    # Apply complex data: P2 stoich becomes 2.
    apply_complex_data(ec_model, apply=True)

    post_coef = next(c for m, c in r.metabolites.items() if m.id == "prot_P2")
    expected_post = -(2 * 20000.0 / (100.0 * 3600.0))
    assert post_coef == pytest.approx(expected_post)


def test_apply_false_leaves_kcat_coefficients_stale():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=True)

    r = ec_model.reactions.get_by_id("R2_EXP_1")
    pre_coef = next(c for m, c in r.metabolites.items() if m.id == "prot_P2")
    apply_complex_data(ec_model, apply=False)
    post_coef = next(c for m, c in r.metabolites.items() if m.id == "prot_P2")

    # Coefficient unchanged because we did not re-apply kcats.
    assert post_coef == pre_coef
    # But rxn_enz_mat was updated.
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 2.0


# --------------------------------------------------------------------------- #
# Stoichiometry edge cases
# --------------------------------------------------------------------------- #

def test_all_zero_stoichiometry_treated_as_all_ones():
    """A complex with all-zero stoichiometry has every entry treated as 1,
    since zero is a placeholder for unknown true counts."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="R2Compl",
        name="x",
        species="t",
        gene_names=["G1", "G2"],
        protein_ids=["P1", "P2"],
        stoichiometry=[0, 0],
    )]
    apply_complex_data(ec_model, complex_data=data, apply=False)
    assert _stoich(ec_model, "R2_EXP_1", "P1") == 1.0
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 1.0


def test_proteins_outside_model_are_ignored():
    """A complex referencing proteins not in ec.enzymes should not crash."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="C_outside",
        name="outside",
        species="t",
        gene_names=["G_x", "G_y"],
        protein_ids=["P_x", "P_y"],  # neither in the model
        stoichiometry=[1, 1],
    )]
    # Should run without error; nothing matches anything.
    apply_complex_data(ec_model, complex_data=data, apply=False)


def test_missing_path_and_no_complex_data_raises(tmp_path):
    """If both path and complex_data are None, the adapter is consulted.
    If the adapter's data folder has no ComplexPortal.json, raise."""
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\nenzyme_comp = "c"\n'
    )
    adapter = ModelAdapter.from_folder(tmp_path)
    ec_model = EcModel("empty", adapter=adapter)

    with pytest.raises(FileNotFoundError):
        apply_complex_data(ec_model)


def test_partial_zero_stoichiometry_counts_the_subunit_as_one():
    """A subunit with stoichiometry 0 has no stated stoichiometry: it is a
    subunit with stoichiometry 1, so the complex still matches."""
    ec_model = _ectestgem_ec_model()
    data = [ComplexPortalEntry(
        complex_id="R2Compl",
        name="x",
        species="t",
        gene_names=["G1", "G2"],
        protein_ids=["P1", "P2"],
        stoichiometry=[2, 0],
    )]
    apply_complex_data(ec_model, complex_data=data, apply=False)
    assert _stoich(ec_model, "R2_EXP_1", "P1") == 2.0
    assert _stoich(ec_model, "R2_EXP_1", "P2") == 1.0
