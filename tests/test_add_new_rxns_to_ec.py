"""Tests for add_new_rxns_to_ec."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import (
    AddNewRxnsResult,
    NewEnzyme,
    add_new_rxns_to_ec,
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path, *, enzyme_comp: str = "c") -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\n'
        f'org_name = "test"\n'
        f'enzyme_comp = "{enzyme_comp}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_base_model(adapter: ModelAdapter) -> EcModel:
    """A minimal ec model with a prot_pool met and one existing enzyme."""
    model = EcModel("base", adapter=adapter)
    A_c = cobra.Metabolite("A_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_X = cobra.Metabolite("prot_X", compartment="c")
    model.add_metabolites([A_c, pool, prot_X])

    usage_X = cobra.Reaction("usage_prot_X")
    usage_X.add_metabolites({pool: -1.0, prot_X: 1.0})
    usage_X.lower_bound = 0.0; usage_X.upper_bound = 1000.0
    usage_X.gene_reaction_rule = "g_X"
    model.add_reactions([usage_X])

    model.ec = EcData(
        rxns=[],
        kcat=np.empty(0),
        source=[],
        notes=[],
        eccodes=[],
        genes=["g_X"],
        enzymes=["X"],
        mw=np.array([100.0]),
        sequence=[""],
        concs=np.array([np.nan]),
        rxn_enz_mat=sparse.csr_matrix((0, 1), dtype=float),
    )
    return model


def _make_rxn(
    rxn_id: str,
    *,
    metabolites: dict,
    gpr: str = "",
    lb: float = 0.0,
    ub: float = 1000.0,
) -> cobra.Reaction:
    rxn = cobra.Reaction(rxn_id)
    rxn.add_metabolites(metabolites)
    rxn.lower_bound = lb
    rxn.upper_bound = ub
    if gpr:
        rxn.gene_reaction_rule = gpr
    return rxn


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_gecko_light_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    model.ec.gecko_light = True
    with pytest.raises(NotImplementedError, match="gecko-light"):
        add_new_rxns_to_ec(model, [], [])


def test_no_adapter_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        add_new_rxns_to_ec(model, [], [])


def test_missing_gene_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])
    rxn = _make_rxn("R_new", metabolites={A_c: -1.0, B_c: 1.0}, gpr="g_unknown")
    with pytest.raises(ValueError, match="missing"):
        add_new_rxns_to_ec(model, [rxn], [])


# --------------------------------------------------------------------------- #
# Single new reaction with new enzyme
# --------------------------------------------------------------------------- #

def test_simple_add(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    new_enz = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    rxn = _make_rxn(
        "R_AY",
        metabolites={A_c: -1.0, B_c: 1.0},
        gpr="g_Y",
    )

    result = add_new_rxns_to_ec(model, [rxn], [new_enz])

    assert isinstance(result, AddNewRxnsResult)
    assert result.enz_added == ["Y"]
    assert result.rxns_added == ["R_AY"]

    # Topology: prot_Y, usage_prot_Y exist.
    assert "prot_Y" in {m.id for m in model.metabolites}
    assert "usage_prot_Y" in {r.id for r in model.reactions}
    # ec.enzymes extended.
    assert "Y" in model.ec.enzymes
    # ec.rxns extended (rxn has GPR).
    assert "R_AY" in model.ec.rxns


# --------------------------------------------------------------------------- #
# Already-present enzymes are skipped with warning
# --------------------------------------------------------------------------- #

def test_existing_enzyme_skipped(tmp_path, caplog):
    import logging
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    # X is already in the base model.
    enz_dup = NewEnzyme(enzyme="X", gene="g_X", mw=999.0)
    enz_new = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    rxn = _make_rxn(
        "R_AY", metabolites={A_c: -1.0, B_c: 1.0}, gpr="g_Y",
    )

    with caplog.at_level(logging.WARNING):
        result = add_new_rxns_to_ec(model, [rxn], [enz_dup, enz_new])

    assert "already in" in caplog.text
    assert result.enz_added == ["Y"]
    # X mw unchanged.
    assert model.ec.mw[model.ec.enzymes.index("X")] == 100.0


# --------------------------------------------------------------------------- #
# Reversibility splitting
# --------------------------------------------------------------------------- #

def test_reversible_rxn_splits_to_forward_and_REV(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    new_enz = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    rxn = _make_rxn(
        "R_AY",
        metabolites={A_c: -1.0, B_c: 1.0},
        gpr="g_Y",
        lb=-500.0, ub=1000.0,
    )

    result = add_new_rxns_to_ec(model, [rxn], [new_enz])

    assert sorted(result.rxns_added) == ["R_AY", "R_AY_REV"]
    # Forward has lb=0; REV has negated stoichiometry and ub=500.
    fwd = model.reactions.get_by_id("R_AY")
    rev = model.reactions.get_by_id("R_AY_REV")
    assert fwd.lower_bound == 0.0
    assert rev.lower_bound == 0.0
    assert rev.upper_bound == 500.0
    assert rev.metabolites[A_c] == 1.0  # negated from -1
    assert rev.metabolites[B_c] == -1.0  # negated from +1


# --------------------------------------------------------------------------- #
# Isozyme splitting
# --------------------------------------------------------------------------- #

def test_isozyme_or_splits_to_EXP_n(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    enz_Y = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    enz_Z = NewEnzyme(enzyme="Z", gene="g_Z", mw=60.0)

    rxn = _make_rxn(
        "R_AY",
        metabolites={A_c: -1.0, B_c: 1.0},
        gpr="g_Y or g_Z",
    )
    result = add_new_rxns_to_ec(model, [rxn], [enz_Y, enz_Z])

    assert sorted(result.rxns_added) == ["R_AY_EXP_1", "R_AY_EXP_2"]
    r1 = model.reactions.get_by_id("R_AY_EXP_1")
    r2 = model.reactions.get_by_id("R_AY_EXP_2")
    # Each variant has exactly one gene (no OR).
    assert "or" not in r1.gene_reaction_rule
    assert "or" not in r2.gene_reaction_rule


def test_combined_reversible_and_isozyme(tmp_path):
    """Reversible with OR -> 4 variants: _EXP_1, _EXP_2, _REV_EXP_1, _REV_EXP_2."""
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    enz_Y = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    enz_Z = NewEnzyme(enzyme="Z", gene="g_Z", mw=60.0)

    rxn = _make_rxn(
        "R_AY",
        metabolites={A_c: -1.0, B_c: 1.0},
        gpr="g_Y or g_Z",
        lb=-500.0,
    )
    result = add_new_rxns_to_ec(model, [rxn], [enz_Y, enz_Z])

    assert set(result.rxns_added) == {
        "R_AY_EXP_1", "R_AY_EXP_2",
        "R_AY_REV_EXP_1", "R_AY_REV_EXP_2",
    }


# --------------------------------------------------------------------------- #
# ec.rxn_enz_mat updated correctly
# --------------------------------------------------------------------------- #

def test_rxn_enz_mat_grows_with_new_rxn_and_enzyme(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    enz_Y = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    rxn = _make_rxn(
        "R_AY", metabolites={A_c: -1.0, B_c: 1.0}, gpr="g_Y",
    )
    add_new_rxns_to_ec(model, [rxn], [enz_Y])

    # 1 ec rxn, 2 enzymes (X, Y).
    assert model.ec.rxn_enz_mat.shape == (1, 2)
    # The new rxn maps to enzyme Y (index 1).
    dense = model.ec.rxn_enz_mat.toarray()
    y_idx = model.ec.enzymes.index("Y")
    assert dense[0, y_idx] == 1.0
    x_idx = model.ec.enzymes.index("X")
    assert dense[0, x_idx] == 0.0


def test_rxn_without_gpr_not_added_to_ec_rxns(tmp_path):
    """A reaction with no GPR is added to the cobra model but not
    to model.ec.rxns."""
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])

    rxn = _make_rxn("R_no_gpr", metabolites={A_c: -1.0, B_c: 1.0})
    add_new_rxns_to_ec(model, [rxn], [])
    assert "R_no_gpr" in {r.id for r in model.reactions}
    assert "R_no_gpr" not in model.ec.rxns


# --------------------------------------------------------------------------- #
# Multiple new reactions
# --------------------------------------------------------------------------- #

def test_multiple_new_rxns_added(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    model.add_metabolites([B_c, C_c])

    enz_Y = NewEnzyme(enzyme="Y", gene="g_Y", mw=50.0)
    enz_Z = NewEnzyme(enzyme="Z", gene="g_Z", mw=60.0)
    r1 = _make_rxn("R_AY", metabolites={A_c: -1.0, B_c: 1.0}, gpr="g_Y")
    r2 = _make_rxn("R_BZ", metabolites={B_c: -1.0, C_c: 1.0}, gpr="g_Z")

    result = add_new_rxns_to_ec(model, [r1, r2], [enz_Y, enz_Z])

    assert sorted(result.rxns_added) == ["R_AY", "R_BZ"]
    assert sorted(result.enz_added) == ["Y", "Z"]
    assert "R_AY" in model.ec.rxns
    assert "R_BZ" in model.ec.rxns


# --------------------------------------------------------------------------- #
# Genes that already have enzymes
# --------------------------------------------------------------------------- #

def test_reaction_with_genes_that_already_have_enzymes_is_added(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_base_model(adapter)
    A_c = model.metabolites.get_by_id("A_c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    model.add_metabolites([B_c])
    n_enzymes = len(model.ec.enzymes)
    rxn = _make_rxn(
        "R_AX", metabolites={A_c: -1.0, B_c: 1.0}, gpr="g_X",
    )

    result = add_new_rxns_to_ec(model, [rxn], [])

    assert result.enz_added == []
    assert len(model.ec.enzymes) == n_enzymes
    assert "R_AX" in model.ec.rxns
