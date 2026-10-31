"""Tests for constrain_enz_concs."""
import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import constrain_enz_concs


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _ec_model_with_pool(
    enzymes: list[str],
    *,
    concs: list[float] | None = None,
    add_pool: bool = True,
    add_usage_rxns: bool = True,
) -> EcModel:
    """Build an EcModel with a prot_pool met and usage_prot_<enzyme>
    reactions, plus a populated ec.concs."""
    g = len(enzymes)
    if concs is None:
        concs = [np.nan] * g

    model = EcModel("test")

    if add_pool:
        pool = cobra.Metabolite("prot_pool", compartment="c")
        pool.name = "prot_pool"
        model.add_metabolites([pool])

    if add_usage_rxns and add_pool:
        for enz in enzymes:
            prot = cobra.Metabolite(f"prot_{enz}", compartment="c")
            prot.name = f"prot_{enz}"
            model.add_metabolites([prot])
            rxn = cobra.Reaction(f"usage_prot_{enz}", name=f"usage_prot_{enz}")
            rxn.lower_bound = 0.0
            rxn.upper_bound = 1000.0
            rxn.add_metabolites({pool: -1.0, prot: 1.0})
            model.add_reactions([rxn])

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
        concs=np.array(concs, dtype=float),
        rxn_enz_mat=sparse.csr_matrix((0, g), dtype=float),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_missing_prot_pool_raises():
    model = _ec_model_with_pool(["P1"], add_pool=False, add_usage_rxns=False)
    with pytest.raises(ValueError, match="prot_pool"):
        constrain_enz_concs(model)


def test_missing_usage_reaction_raises():
    """A model without usage_prot_<enzyme> rxns (e.g. gecko_light) must raise."""
    model = _ec_model_with_pool(["P1"], add_pool=True, add_usage_rxns=False)
    with pytest.raises(ValueError, match="Usage reaction"):
        constrain_enz_concs(model)


# --------------------------------------------------------------------------- #
# Default behavior
# --------------------------------------------------------------------------- #

def test_nan_conc_keeps_default_ub():
    model = _ec_model_with_pool(["P1"], concs=[np.nan])
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 1000.0


def test_numeric_conc_sets_ub():
    model = _ec_model_with_pool(["P1"], concs=[3.5])
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 3.5


def test_mixed_concs_handled_per_enzyme():
    model = _ec_model_with_pool(
        ["P1", "P2", "P3"],
        concs=[5.0, np.nan, 7.5],
    )
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 5.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 1000.0
    assert model.reactions.get_by_id("usage_prot_P3").upper_bound == 7.5


def test_default_lb_unchanged():
    """The lower bound should remain at 0 (geckopy forward direction)."""
    model = _ec_model_with_pool(["P1"], concs=[5.0])
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").lower_bound == 0.0


def test_stoichiometry_unchanged():
    model = _ec_model_with_pool(["P1"], concs=[5.0])
    constrain_enz_concs(model)
    rxn = model.reactions.get_by_id("usage_prot_P1")
    stoich = {m.id: c for m, c in rxn.metabolites.items()}
    assert stoich == {"prot_pool": -1.0, "prot_P1": 1.0}


# --------------------------------------------------------------------------- #
# Re-application: existing constraints reset
# --------------------------------------------------------------------------- #

def test_re_run_with_new_concs_overwrites_previous():
    model = _ec_model_with_pool(["P1"], concs=[3.0])
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 3.0

    model.ec.concs[0] = 8.0
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 8.0


def test_re_run_with_nan_clears_previous_constraint():
    """A previously-constrained enzyme whose conc becomes NaN should
    have its ub reset to 1000."""
    model = _ec_model_with_pool(["P1"], concs=[3.0])
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 3.0

    model.ec.concs[0] = np.nan
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 1000.0


# --------------------------------------------------------------------------- #
# remove_constraints
# --------------------------------------------------------------------------- #

def test_remove_constraints_resets_all_to_default():
    model = _ec_model_with_pool(
        ["P1", "P2"], concs=[5.0, 7.0],
    )
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 5.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 7.0

    constrain_enz_concs(model, remove_constraints=True)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 1000.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 1000.0


def test_remove_constraints_does_not_modify_concs():
    model = _ec_model_with_pool(["P1"], concs=[5.0])
    constrain_enz_concs(model, remove_constraints=True)
    assert model.ec.concs[0] == 5.0


def test_restrict_to_only_updates_listed_enzymes():
    """With ``restrict_to``, only the listed enzymes' usage reactions
    are touched; others keep whatever bounds they had."""
    model = _ec_model_with_pool(["P1", "P2"], concs=[5.0, 7.0])
    # Pre-set P2's bound to a sentinel; constrain only P1.
    model.reactions.get_by_id("usage_prot_P2").upper_bound = 42.0
    constrain_enz_concs(model, restrict_to=["P1"])
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 5.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 42.0


def test_restrict_to_unknown_ids_ignored():
    """Unknown uniprot IDs in ``restrict_to`` are silently skipped."""
    model = _ec_model_with_pool(["P1"], concs=[5.0])
    # Nothing in restrict_to matches the model -> no-op.
    model.reactions.get_by_id("usage_prot_P1").upper_bound = 99.0
    constrain_enz_concs(model, restrict_to=["P_ghost"])
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 99.0


# --------------------------------------------------------------------------- #
# Empty model
# --------------------------------------------------------------------------- #

def test_no_enzymes_no_op():
    model = _ec_model_with_pool([])
    constrain_enz_concs(model)  # should not raise


# --------------------------------------------------------------------------- #
# Integration with fill_enz_concs
# --------------------------------------------------------------------------- #

def test_integration_with_fill_enz_concs():
    from geckopy.databases import ProtData
    from geckopy.limit_proteins import fill_enz_concs

    model = _ec_model_with_pool(["P1", "P2", "P3"])
    pd = ProtData(
        uniprot_ids=["P1", "P3"],
        abundances=np.array([4.0, 9.0]),
    )
    fill_enz_concs(model, pd)
    constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 4.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 1000.0
    assert model.reactions.get_by_id("usage_prot_P3").upper_bound == 9.0


# --------------------------------------------------------------------------- #
# Negative concentration
# --------------------------------------------------------------------------- #

def test_negative_conc_raises_before_changing_the_model():
    model = _ec_model_with_pool(["P1", "P2"], concs=[3.0, -0.5])
    with pytest.raises(ValueError, match="negative concentration"):
        constrain_enz_concs(model)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 1000.0
    assert model.reactions.get_by_id("usage_prot_P2").upper_bound == 1000.0


def test_negative_conc_is_not_read_when_removing_constraints():
    model = _ec_model_with_pool(["P1"], concs=[-0.5])
    constrain_enz_concs(model, remove_constraints=True)
    assert model.reactions.get_by_id("usage_prot_P1").upper_bound == 1000.0


# --------------------------------------------------------------------------- #
# Prior bounds and coefficient
# --------------------------------------------------------------------------- #

def test_prior_lower_bound_above_the_new_upper_bound_is_reset():
    model = _ec_model_with_pool(["P1"], concs=[0.01])
    rxn = model.reactions.get_by_id("usage_prot_P1")
    rxn.lower_bound = 0.05
    pool = model.metabolites.get_by_id("prot_pool")
    rxn.add_metabolites({pool: -2.0}, combine=False)
    constrain_enz_concs(model)
    assert rxn.bounds == (0.0, 0.01)
    assert rxn.metabolites[pool] == -1.0
