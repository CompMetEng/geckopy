"""Behaviour shared with GECKO MATLAB: EC cell separators, zero-bound capacity,
the target growth of flexibilize_enz_concs and isozyme competition in gecko-light."""
import copy
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.ec_data import EcData
from geckopy.ec_model.pipeline import apply_kcat_constraints
from geckopy.get_enzyme_data import find_ec_in_db
from geckopy.limit_proteins import flexibilize_enz_concs
from geckopy.utilities import enzyme_usage, report_enzyme_usage

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


# --------------------------------------------------------------------------- #
# find_ec_in_db: a database cell may separate its codes with ';', spaces or both
# --------------------------------------------------------------------------- #

def _db(rows):
    codes = [ec for _, ec, _ in rows]
    mw = np.array([m for _, _, m in rows], dtype=float)
    index: dict[str, list[int]] = {}
    for i, (gene, _, _) in enumerate(rows):
        index.setdefault(gene, []).append(i)
    return codes, mw, index


def test_semicolon_space_cell_intersects_with_partner_gene():
    codes, mw, index = _db([("g1", "1.1.1.1; 2.2.2.2", 10.0), ("g2", "2.2.2.2", 10.0)])
    assert find_ec_in_db(["g1", "g2"], codes, mw, index) == "2.2.2.2"


def test_space_separated_cell_is_split():
    codes, mw, index = _db([("g1", "1.1.1.4 1.1.1.303", 10.0)])
    assert find_ec_in_db(["g1"], codes, mw, index) == "1.1.1.4;1.1.1.303"


def test_semicolon_without_space_is_split():
    codes, mw, index = _db([("g1", "1.1.1.4;1.1.1.303", 10.0)])
    assert find_ec_in_db(["g1"], codes, mw, index) == "1.1.1.4;1.1.1.303"


def test_space_separated_wildcard_cell_drops_the_parent_wildcard():
    codes, mw, index = _db([("g1", "1.1.1.4 1.1.1.- 1.1.1.303", 10.0)])
    assert find_ec_in_db(["g1"], codes, mw, index) == "1.1.1.4;1.1.1.303"


# --------------------------------------------------------------------------- #
# enzyme_usage: a zero usage upper bound gives NaN, never 0
# --------------------------------------------------------------------------- #

def _toy_usage(enzyme_ub: float) -> EcModel:
    model = EcModel("toy")
    a_e = cobra.Metabolite("A_e", compartment="e")
    a_c = cobra.Metabolite("A_c", compartment="c")
    b_c = cobra.Metabolite("B_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_e = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([a_e, a_c, b_c, pool, prot_e])
    ex_a = cobra.Reaction("EX_A")
    ex_a.add_metabolites({a_e: -1.0})
    ex_a.bounds = (-1000.0, 0.0)
    tr_a = cobra.Reaction("TR_A")
    tr_a.add_metabolites({a_e: -1.0, a_c: 1.0})
    tr_a.bounds = (0.0, 1000.0)
    r_ab = cobra.Reaction("R_AB")
    r_ab.add_metabolites({a_c: -1.0, b_c: 1.0, prot_e: -1.0})
    r_ab.bounds = (0.0, 1000.0)
    sk_b = cobra.Reaction("SK_B")
    sk_b.add_metabolites({b_c: -1.0})
    sk_b.bounds = (0.0, 1000.0)
    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.bounds = (0.0, 1000.0)
    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_e: 1.0})
    usage.bounds = (0.0, enzyme_ub)
    model.add_reactions([ex_a, tr_a, r_ab, sk_b, pool_ex, usage])
    model.objective = "SK_B"
    model.ec = EcData(
        rxns=["R_AB"], kcat=np.array([1.0]), source=[""], notes=[""], eccodes=[""],
        genes=["g_E"], enzymes=["E"], mw=np.array([100.0]), sequence=[""],
        concs=np.array([float(enzyme_ub)]), rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


def test_zero_upper_bound_gives_nan_capacity():
    model = _toy_usage(0.0)
    result = enzyme_usage(model, model.optimize().fluxes)
    assert np.isnan(result.cap_usage[0])
    assert result.abs_usage[0] == 0.0


def test_positive_upper_bound_gives_the_ratio():
    model = _toy_usage(5.0)
    result = enzyme_usage(model, model.optimize().fluxes)
    assert result.cap_usage[0] == pytest.approx(1.0)


def test_report_lists_no_zero_bound_enzyme_as_high_capacity():
    model = _toy_usage(0.0)
    usage = enzyme_usage(model, model.optimize().fluxes)
    report = report_enzyme_usage(model, usage)
    assert len(report.high_cap_usage) == 0


# --------------------------------------------------------------------------- #
# flexibilize_enz_concs: when no enzyme limits growth any more, the growth reached
# is the target of the final pass
# --------------------------------------------------------------------------- #

def _flex_toy(tmp_path: Path, *, uptake_cap: float) -> EcModel:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\ngr_exp = 5.0\nbio_rxn = "biomass"\n'
    )
    adapter = ModelAdapter.from_folder(tmp_path)
    model = EcModel("toy", adapter=adapter)
    a_e = cobra.Metabolite("A_e", compartment="e")
    a_c = cobra.Metabolite("A_c", compartment="c")
    b_c = cobra.Metabolite("B_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_e = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([a_e, a_c, b_c, pool, prot_e])
    ex_a = cobra.Reaction("EX_A")
    ex_a.add_metabolites({a_e: -1.0})
    ex_a.bounds = (-1000.0, 0.0)
    tr_a = cobra.Reaction("TR_A")
    tr_a.add_metabolites({a_e: -1.0, a_c: 1.0})
    tr_a.bounds = (0.0, uptake_cap)
    r_ab = cobra.Reaction("R_AB")
    r_ab.add_metabolites({a_c: -1.0, b_c: 1.0, prot_e: -1.0})
    r_ab.bounds = (0.0, 1000.0)
    bio = cobra.Reaction("biomass")
    bio.add_metabolites({b_c: -1.0})
    bio.bounds = (0.0, 1000.0)
    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.bounds = (0.0, 1000.0)
    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_e: 1.0})
    usage.bounds = (0.0, 1.0)
    model.add_reactions([ex_a, tr_a, r_ab, bio, pool_ex, usage])
    model.objective = "biomass"
    model.ec = EcData(
        rxns=["R_AB"], kcat=np.array([1.0]), source=[""], notes=[""], eccodes=[""],
        genes=["g_E"], enzymes=["E"], mw=np.array([100.0]), sequence=[""],
        concs=np.array([1.0]), rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


def test_growth_capped_below_target_is_the_final_target(tmp_path):
    model = _flex_toy(tmp_path, uptake_cap=2.5)
    result = flexibilize_enz_concs(model)
    assert result.uniprot_ids == ["E"]
    upper = model.reactions.get_by_id("usage_prot_E").upper_bound
    # The band's low edge below the reached growth 2.5, not an infeasible 5.0.
    assert np.isfinite(upper)
    assert upper == pytest.approx(2.5 * (1 - 0.25 / 100), rel=1e-3)
    assert result.flex_concs[0] == pytest.approx(upper)


# --------------------------------------------------------------------------- #
# apply_kcat_constraints, gecko-light: naming one isozyme row updates the whole
# reaction, so every isozyme of it competes for the cheapest cost
# --------------------------------------------------------------------------- #

_LIGHT_CACHE: EcModel | None = None


def _light_model() -> EcModel:
    global _LIGHT_CACHE
    if _LIGHT_CACHE is None:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        gem = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _LIGHT_CACHE = make_ec_model(gem, adapter, gecko_light=True)
    return copy.deepcopy(_LIGHT_CACHE)


def _pool_coef(model: EcModel, rxn_id: str) -> float:
    rxn = model.reactions.get_by_id(rxn_id)
    for met, coef in rxn.metabolites.items():
        if met.id == "prot_pool":
            return float(coef)
    return 0.0


def _row_cost(model: EcModel, row: int) -> float:
    csr = model.ec.rxn_enz_mat.tocsr().getrow(row)
    return float(np.sum(csr.data * model.ec.mw[csr.indices])) / float(model.ec.kcat[row])


def test_naming_the_worse_isozyme_keeps_the_cheapest_cost_of_the_reaction():
    model = _light_model()
    complex_row = model.ec.rxns.index("001_R2")
    single_row = model.ec.rxns.index("002_R2")
    model.ec.kcat[complex_row] = 50.0
    model.ec.kcat[single_row] = 50.0
    apply_kcat_constraints(model)
    # The single-subunit isozyme is the cheaper one; make it worse and name only it.
    model.ec.kcat[single_row] = 0.5
    cheapest = min(_row_cost(model, complex_row), _row_cost(model, single_row))
    assert _row_cost(model, complex_row) < _row_cost(model, single_row)

    apply_kcat_constraints(model, update_rxns=["002_R2"])

    assert _pool_coef(model, "R2") == pytest.approx(-cheapest / 3600.0, rel=1e-12)


def test_naming_one_isozyme_leaves_other_reactions_untouched():
    model = _light_model()
    for rxn_id in ("001_R2", "002_R2", "001_R3"):
        model.ec.kcat[model.ec.rxns.index(rxn_id)] = 10.0
    apply_kcat_constraints(model, update_rxns=["001_R3"])
    assert _pool_coef(model, "R2") == 0.0
    assert _pool_coef(model, "R3") < 0.0


def test_isozyme_row_without_enzymes_is_skipped():
    model = _light_model()
    complex_row = model.ec.rxns.index("001_R2")
    single_row = model.ec.rxns.index("002_R2")
    model.ec.kcat[complex_row] = 50.0
    model.ec.kcat[single_row] = 50.0
    matrix = model.ec.rxn_enz_mat.tolil()
    matrix[single_row, :] = 0.0
    model.ec.rxn_enz_mat = matrix.tocsr()
    apply_kcat_constraints(model, update_rxns=["001_R2"])
    assert _pool_coef(model, "R2") == pytest.approx(-_row_cost(model, complex_row) / 3600.0, rel=1e-12)
