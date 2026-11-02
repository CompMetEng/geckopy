"""Tests for report_enzyme_usage."""
import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import (
    EnzymeUsageReport,
    enzyme_usage,
    report_enzyme_usage,
)


# --------------------------------------------------------------------------- #
# Single-enzyme single-rxn fixture
# --------------------------------------------------------------------------- #

def _build_single_enzyme_model(*, enzyme_ub: float = 5.0) -> EcModel:
    """Builds a toy model with one enzyme (E) catalysing A -> B, wired
    through the protein-pool/usage-reaction machinery, with enzyme_ub
    capping the usage reaction's flux."""
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
    R.name = "A to B"
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
    pool_ex.upper_bound = 100.0

    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage.lower_bound = 0.0
    usage.upper_bound = enzyme_ub

    model.add_reactions([EX_A, TR_A, R, SK_B, pool_ex, usage])
    model.objective = "SK_B"

    model.ec = EcData(
        rxns=["R_AB"],
        kcat=np.array([1.0]),
        source=["initial"],
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
# Single-enzyme multi-rxn fixture (for combined-row testing)
# --------------------------------------------------------------------------- #

def _build_multi_rxn_per_enzyme_model() -> EcModel:
    """Enzyme E catalyses two reactions (A->B and A->C). Both
    reactions consume A and prot_E."""
    model = EcModel("toy")

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([A_e, A_c, B_c, C_c, pool, prot_E])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -1000.0
    EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0
    TR_A.upper_bound = 1000.0

    R_AB = cobra.Reaction("R_AB")
    R_AB.add_metabolites({A_c: -1.0, B_c: 1.0, prot_E: -1.0})
    R_AB.lower_bound = 0.0; R_AB.upper_bound = 1000.0

    R_AC = cobra.Reaction("R_AC")
    R_AC.add_metabolites({A_c: -1.0, C_c: 1.0, prot_E: -2.0})
    R_AC.lower_bound = 0.0; R_AC.upper_bound = 1000.0

    SK_B = cobra.Reaction("SK_B")
    SK_B.add_metabolites({B_c: -1.0})
    SK_B.lower_bound = 0.0; SK_B.upper_bound = 1000.0

    SK_C = cobra.Reaction("SK_C")
    SK_C.add_metabolites({C_c: -1.0})
    SK_C.lower_bound = 0.0; SK_C.upper_bound = 1000.0

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({B_c: -1.0, C_c: -1.0})
    BIO.lower_bound = 0.0; BIO.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0; pool_ex.upper_bound = 100.0

    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage.lower_bound = 0.0; usage.upper_bound = 100.0

    model.add_reactions(
        [EX_A, TR_A, R_AB, R_AC, SK_B, SK_C, BIO, pool_ex, usage]
    )
    model.objective = "biomass"  # forces BOTH reactions to carry flux

    model.ec = EcData(
        rxns=["R_AB", "R_AC"],
        kcat=np.array([1.0, 0.5]),
        source=["initial", "initial"],
        notes=["", ""],
        eccodes=["", ""],
        genes=["g_E"],
        enzymes=["E"],
        mw=np.array([100.0]),
        sequence=[""],
        concs=np.array([100.0]),
        rxn_enz_mat=sparse.csr_matrix([[1.0], [1.0]]),
    )
    return model


# --------------------------------------------------------------------------- #
# Single-enzyme single-rxn cases
# --------------------------------------------------------------------------- #

def test_returns_report_dataclass():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud)
    assert isinstance(rep, EnzymeUsageReport)
    assert rep.total_usage_flux == 100.0


def test_high_cap_usage_includes_at_capacity_enzyme():
    model = _build_single_enzyme_model(enzyme_ub=5.0)
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud)
    assert "E" in list(rep.high_cap_usage["prot_id"])
    row = rep.high_cap_usage[
        rep.high_cap_usage["prot_id"] == "E"
    ].iloc[0]
    assert row["cap_usage"] == pytest.approx(1.0)


def test_high_cap_usage_excludes_low_capacity():
    model = _build_single_enzyme_model(enzyme_ub=5.0)
    model.reactions.get_by_id("EX_A").lower_bound = -0.5
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    # cap_usage = 0.5/5 = 0.1, below threshold 0.9.
    rep = report_enzyme_usage(model, ud, high_cap_usage=0.9)
    assert "E" not in list(rep.high_cap_usage["prot_id"])


def test_top_abs_usage_basic():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud, top_abs_usage=10)
    assert len(rep.top_abs_usage) == 1
    assert rep.top_abs_usage.iloc[0]["prot_id"] == "E"


def test_top_abs_usage_perc_relative_to_pool():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud)
    # Usage = 5; pool ub = 100 -> perc = 5%.
    row = rep.top_abs_usage.iloc[0]
    assert row["abs_usage"] == pytest.approx(5.0)
    assert row["perc_usage"] == pytest.approx(5.0)


def test_columns_present_in_both_reports():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud)
    assert list(rep.high_cap_usage.columns) == [
        "prot_id", "gene_id", "abs_usage", "cap_usage",
        "kcat", "source", "rxn_id", "rxn_name", "gr_rule",
    ]
    assert list(rep.top_abs_usage.columns) == [
        "prot_id", "gene_id", "abs_usage", "perc_usage",
        "kcat", "source", "rxn_id", "rxn_name", "gr_rule",
    ]


def test_kcat_and_source_populated_from_ec():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud)
    row = rep.top_abs_usage.iloc[0]
    assert row["kcat"] == pytest.approx(1.0)
    assert row["source"] == "initial"
    assert row["rxn_id"] == "R_AB"
    assert row["rxn_name"] == "A to B"


# --------------------------------------------------------------------------- #
# Multi-rxn-per-enzyme: combined header + detail rows
# --------------------------------------------------------------------------- #

def test_multi_rxn_emits_header_plus_detail_rows():
    model = _build_multi_rxn_per_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud, top_abs_usage=10)
    enzyme_rows = rep.top_abs_usage[
        rep.top_abs_usage["prot_id"] == "E"
    ]
    assert len(enzyme_rows) == 3  # header + 2 detail rows
    header = enzyme_rows.iloc[0]
    assert header["rxn_id"] == "==="
    assert header["source"] == "==="
    assert header["gr_rule"] == "==="
    assert "involved in multiple rxns" in header["rxn_name"]
    detail_rxn_ids = list(enzyme_rows.iloc[1:]["rxn_id"])
    assert sorted(detail_rxn_ids) == ["R_AB", "R_AC"]


def test_multi_rxn_detail_abs_usage_proportional_to_stoichiometry():
    """R_AB consumes 1*prot_E, R_AC consumes 2*prot_E. With equal
    flux through both (driven by biomass), the detail abs_usage is
    proportional to stoichiometry."""
    model = _build_multi_rxn_per_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud, top_abs_usage=10)
    detail_rows = rep.top_abs_usage[
        (rep.top_abs_usage["prot_id"] == "E")
        & (rep.top_abs_usage["rxn_id"] != "===")
    ]
    abs_by_rxn = dict(zip(detail_rows["rxn_id"], detail_rows["abs_usage"]))
    # R_AB consumes 1*flux; R_AC consumes 2*flux. With equal flux,
    # ratio R_AC : R_AB = 2 : 1.
    assert abs_by_rxn["R_AC"] == pytest.approx(2 * abs_by_rxn["R_AB"])


def test_multi_rxn_high_cap_detail_cap_usage_sums_to_total():
    """The proportional cap_usage across detail rows should sum to
    the original total cap_usage of the enzyme."""
    model = _build_multi_rxn_per_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    # Lower threshold so the enzyme qualifies.
    rep = report_enzyme_usage(model, ud, high_cap_usage=0.0)
    enzyme_rows = rep.high_cap_usage[
        rep.high_cap_usage["prot_id"] == "E"
    ]
    if len(enzyme_rows) <= 1:
        pytest.skip("Multi-rxn breakdown not triggered.")
    header = enzyme_rows.iloc[0]
    detail_sum = enzyme_rows.iloc[1:]["cap_usage"].sum()
    assert detail_sum == pytest.approx(header["cap_usage"], rel=1e-6)


# --------------------------------------------------------------------------- #
# top_abs_usage = 0 / inf -> all enzymes
# --------------------------------------------------------------------------- #

def test_top_abs_usage_zero_returns_all():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud, top_abs_usage=0)
    assert len(rep.top_abs_usage) == 1


def test_top_abs_usage_inf_returns_all():
    model = _build_single_enzyme_model()
    sol = model.optimize()
    ud = enzyme_usage(model, sol.fluxes)
    rep = report_enzyme_usage(model, ud, top_abs_usage=float("inf"))
    assert len(rep.top_abs_usage) == 1


# --------------------------------------------------------------------------- #
# Invalid top_abs_usage
# --------------------------------------------------------------------------- #

def test_negative_top_abs_usage_raises():
    model = _build_single_enzyme_model()
    ud = enzyme_usage(model, model.optimize().fluxes)
    with pytest.raises(ValueError, match="top_abs_usage"):
        report_enzyme_usage(model, ud, top_abs_usage=-2)
