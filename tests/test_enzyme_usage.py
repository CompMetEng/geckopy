"""Tests for enzyme_usage."""
import cobra
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import EnzymeUsageResult, enzyme_usage


# --------------------------------------------------------------------------- #
# Tiny enzyme-constrained model fixture
# --------------------------------------------------------------------------- #

def _build_toy(*, enzyme_ub: float = 5.0) -> EcModel:
    """Tiny enzyme-constrained model. Same shape as in
    test_get_conc_control_coeffs."""
    model = EcModel("toy")

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([A_e, A_c, B_c, pool, prot_E])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -1000.0
    EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0
    TR_A.upper_bound = 1000.0

    R = cobra.Reaction("R_AB")
    R.add_metabolites({A_c: -1.0, B_c: 1.0, prot_E: -1.0})
    R.lower_bound = 0.0
    R.upper_bound = 1000.0

    SK_B = cobra.Reaction("SK_B")
    SK_B.add_metabolites({B_c: -1.0})
    SK_B.lower_bound = 0.0
    SK_B.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0
    pool_ex.upper_bound = 1000.0

    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage.lower_bound = 0.0
    usage.upper_bound = enzyme_ub

    model.add_reactions([EX_A, TR_A, R, SK_B, pool_ex, usage])
    model.objective = "SK_B"

    model.ec = EcData(
        rxns=["R_AB"],
        kcat=np.array([1.0]),
        source=[""],
        notes=[""],
        eccodes=[""],
        genes=["g_E"],
        enzymes=["E"],
        mw=np.array([100.0]),
        sequence=[""],
        concs=np.array([float(enzyme_ub)]),
        rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_gecko_light_raises():
    model = _build_toy()
    model.ec.gecko_light = True
    with pytest.raises(NotImplementedError, match="gecko-light"):
        enzyme_usage(model, pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_basic_usage_at_optimum():
    """At the LP optimum, R_AB flux = 5 (= enzyme ub), so usage_prot_E
    flux = 5, cap usage = 5/5 = 1.0."""
    model = _build_toy(enzyme_ub=5.0)
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes)

    assert isinstance(result, EnzymeUsageResult)
    assert result.prot_id == ["E"]
    assert result.abs_usage[0] == pytest.approx(5.0)
    assert result.ub[0] == pytest.approx(5.0)
    assert result.cap_usage[0] == pytest.approx(1.0)


def test_partial_usage_when_not_at_capacity():
    """Cap upstream so the enzyme runs at half capacity."""
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = -2.5
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes)
    assert result.abs_usage[0] == pytest.approx(2.5)
    assert result.cap_usage[0] == pytest.approx(0.5)


def test_zero_usage_when_no_flux():
    """When no flux flows, usage and cap are both 0."""
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = 0.0  # no uptake
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes)
    assert result.abs_usage[0] == 0.0
    assert result.cap_usage[0] == 0.0


# --------------------------------------------------------------------------- #
# include_zero
# --------------------------------------------------------------------------- #

def test_include_zero_false_drops_unused_enzymes():
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = 0.0
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes, include_zero=False)
    assert result.prot_id == []
    assert result.abs_usage.size == 0
    assert result.cap_usage.size == 0


def test_include_zero_true_keeps_unused_enzymes():
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = 0.0
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes, include_zero=True)
    assert result.prot_id == ["E"]


# --------------------------------------------------------------------------- #
# Flux input shape
# --------------------------------------------------------------------------- #

def test_dict_fluxes_supported():
    model = _build_toy(enzyme_ub=5.0)
    fluxes = {"usage_prot_E": 3.0}
    result = enzyme_usage(model, fluxes)
    assert result.abs_usage[0] == 3.0
    assert result.cap_usage[0] == pytest.approx(0.6)


def test_missing_rxn_in_fluxes_treated_as_zero():
    model = _build_toy(enzyme_ub=5.0)
    fluxes = {}  # no entries
    result = enzyme_usage(model, fluxes)
    assert result.abs_usage[0] == 0.0
    assert result.cap_usage[0] == 0.0


# --------------------------------------------------------------------------- #
# Zero upper bound -> cap_usage = NaN
# --------------------------------------------------------------------------- #

def test_zero_ub_cap_usage_is_nan():
    model = _build_toy(enzyme_ub=0.0)
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes)
    # ub=0 -> no growth -> abs_usage=0; the ratio is undefined, as for Enzyme.cap_usage.
    assert result.abs_usage[0] == 0.0
    assert np.isnan(result.cap_usage[0])


# --------------------------------------------------------------------------- #
# fluxes carried through
# --------------------------------------------------------------------------- #

def test_fluxes_carried_through_to_result():
    model = _build_toy()
    sol = model.optimize()
    result = enzyme_usage(model, sol.fluxes)
    # Same Series object (or equivalent values).
    assert result.fluxes is sol.fluxes
