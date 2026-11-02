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


# --------------------------------------------------------------------------- #
# Enzyme without a usage reaction
# --------------------------------------------------------------------------- #

def test_enzyme_without_usage_reaction_is_left_out_with_a_warning(caplog):
    model = _build_toy(enzyme_ub=5.0)
    model.ec.enzymes.append("F")
    with caplog.at_level("WARNING"):
        result = enzyme_usage(model, {"usage_prot_E": 3.0})
    assert result.prot_id == ["E"]
    assert "no usage reaction for the enzymes F" in caplog.text


# --------------------------------------------------------------------------- #
# Demand ratio
# --------------------------------------------------------------------------- #

def test_demand_ratio_at_the_measurement():
    """The enzyme runs at its measured concentration: ratio 1."""
    model = _build_toy(enzyme_ub=5.0)
    result = enzyme_usage(model, model.optimize().fluxes)
    assert result.measured_protein == pytest.approx(5.0)
    assert result.demanded_protein == pytest.approx(5.0)
    assert result.demand_ratio == pytest.approx(1.0)
    assert result.demand_ratio_active == pytest.approx(1.0)


def test_demand_ratio_above_one_when_the_bound_is_relaxed():
    """A usage bound above the measurement lets the solution use more of the
    enzyme than was measured."""
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("usage_prot_E").upper_bound = 10.0
    result = enzyme_usage(model, model.optimize().fluxes)
    assert result.abs_usage[0] == pytest.approx(10.0)
    assert result.measured_protein == pytest.approx(5.0)
    assert result.demand_ratio == pytest.approx(2.0)
    assert result.demand_ratio_active == pytest.approx(2.0)


def test_demand_ratio_of_an_inactive_measured_enzyme():
    """Without flux the demand is 0, and no active enzyme is measured."""
    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = 0.0
    result = enzyme_usage(model, model.optimize().fluxes)
    assert result.measured_protein == pytest.approx(5.0)
    assert result.demand_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.measured_protein_active == 0.0
    assert np.isnan(result.demand_ratio_active)


def test_demand_ratio_is_nan_without_a_measured_concentration():
    """An enzyme with a NaN concentration draws from the protein pool and is
    not counted."""
    model = _build_toy(enzyme_ub=5.0)
    model.ec.concs[:] = np.nan
    result = enzyme_usage(model, model.optimize().fluxes)
    assert result.measured_protein == 0.0
    assert result.demanded_protein == 0.0
    assert np.isnan(result.demand_ratio)
    assert np.isnan(result.demand_ratio_active)


def test_demand_ratio_does_not_depend_on_include_zero():
    model, fluxes = _build_mixed()
    fluxes["usage_prot_F"] = 0.0
    with_zero = enzyme_usage(model, fluxes)
    without_zero = enzyme_usage(model, fluxes, include_zero=False)
    assert with_zero.prot_id == ["E", "F", "G"]
    assert without_zero.prot_id == ["E", "G"]
    for name in (
        "measured_protein", "measured_protein_active", "demanded_protein",
        "demand_ratio", "demand_ratio_active",
    ):
        assert getattr(without_zero, name) == getattr(with_zero, name)
    assert without_zero.measured_protein == pytest.approx(8.0)
    assert without_zero.demand_ratio == pytest.approx(0.25)


def test_report_carries_the_demand_ratios():
    from geckopy.utilities import report_enzyme_usage

    model = _build_toy(enzyme_ub=5.0)
    model.reactions.get_by_id("usage_prot_E").upper_bound = 10.0
    result = enzyme_usage(model, model.optimize().fluxes)
    report = report_enzyme_usage(model, result)
    assert report.demand_ratio == pytest.approx(2.0)
    assert report.demand_ratio_active == pytest.approx(2.0)


def _build_mixed():
    """Enzymes E (measured 5), H (measured 4, no usage reaction), F (measured
    3) and G (not measured), with usage fluxes 2, 1e-8 and 7."""
    model = _build_toy(enzyme_ub=5.0)
    for name in ("F", "G"):
        usage = cobra.Reaction(f"usage_prot_{name}")
        usage.upper_bound = 10.0
        model.add_reactions([usage])
    model.ec.enzymes[:] = ["E", "H", "F", "G"]
    model.ec.concs = np.array([5.0, 4.0, 3.0, np.nan])
    fluxes = {"usage_prot_E": 2.0, "usage_prot_F": 1e-8, "usage_prot_G": 7.0}
    return model, fluxes


def test_demand_ratio_over_measured_and_active_enzymes():
    model, fluxes = _build_mixed()
    result = enzyme_usage(model, fluxes)
    assert result.prot_id == ["E", "F", "G"]
    assert result.measured_protein == pytest.approx(8.0)
    assert result.demanded_protein == pytest.approx(2.0 + 1e-8)
    assert result.demand_ratio == pytest.approx((2.0 + 1e-8) / 8.0)
    assert result.measured_protein_active == pytest.approx(5.0)
    assert result.demand_ratio_active == pytest.approx(0.4)


def test_demand_ratio_with_a_short_concentration_vector():
    model, fluxes = _build_mixed()
    model.ec.concs = np.array([5.0])
    result = enzyme_usage(model, fluxes)
    assert result.measured_protein == pytest.approx(5.0)
    assert result.demand_ratio_active == pytest.approx(0.4)
    model.ec.concs = np.array([])
    result = enzyme_usage(model, fluxes)
    assert np.isnan(result.demand_ratio)


def test_report_carries_nan_demand_ratios():
    from geckopy.utilities import report_enzyme_usage

    model = _build_toy(enzyme_ub=5.0)
    model.ec.concs[:] = np.nan
    result = enzyme_usage(model, model.optimize().fluxes)
    report = report_enzyme_usage(model, result)
    assert np.isnan(report.demand_ratio)
    assert np.isnan(report.demand_ratio_active)
