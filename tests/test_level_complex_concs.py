"""Tests for complex_families, level_complex_concs and flexibilize_enz_concs(per_complex=True)."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import (
    complex_families,
    flexibilize_enz_concs,
    level_complex_abundances,
    level_complex_concs,
)

ENZYMES = ["A", "B", "C", "D", "F", "S", "E", "G", "H"]
MW = np.array([100.0, 200.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])


def _ec_only(concs, rows):
    """An EcModel carrying only ec data: ``rows`` is {rxn: {enzyme: copies}}."""
    model = EcModel("toy")
    matrix = np.zeros((len(rows), len(ENZYMES)))
    for r, enzymes in enumerate(rows.values()):
        for enzyme, copies in enzymes.items():
            matrix[r, ENZYMES.index(enzyme)] = copies
    n = len(rows)
    model.ec = EcData(
        rxns=list(rows), kcat=np.ones(n), source=[""] * n, notes=[""] * n, eccodes=[""] * n,
        genes=[f"g_{e}" for e in ENZYMES], enzymes=list(ENZYMES), mw=MW.copy(),
        sequence=[""] * len(ENZYMES), concs=np.asarray(concs, dtype=float),
        rxn_enz_mat=sparse.csr_matrix(matrix),
    )
    return model


ROWS = {
    "R1_EXP_1": {"A": 1, "B": 2},
    "R1_EXP_2": {"A": 1, "B": 2, "C": 1},  # isozyme variant: same complex
    "R2": {"D": 1, "F": 1, "S": 1},
    "R3": {"E": 1, "G": 1, "S": 1},       # S is shared with R2's complex
    "R4": {"H": 1},                       # a single enzyme: not a complex
}


def test_isozyme_variants_are_one_complex_and_a_subunit_in_two_is_shared():
    families = complex_families(_ec_only(np.full(len(ENZYMES), np.nan), ROWS))
    complexes = sorted(sorted(c) for c in families.complexes)
    assert complexes == [["A", "B", "C"], ["D", "F", "S"], ["E", "G", "S"]]
    assert families.shared == {"S"}
    assert sorted(families.members("B")) == ["A", "B", "C"]
    assert families.members("S") == ["S"]
    assert families.members("H") == ["H"]


def test_an_enzyme_that_also_works_alone_is_shared():
    rows = dict(ROWS, R5={"A": 1})
    families = complex_families(_ec_only(np.full(len(ENZYMES), np.nan), rows))
    assert "A" in families.shared


def test_levelling_takes_the_median_per_copy_and_scales_by_stoichiometry():
    # molar per copy: A = 1, B = 4 / 2 = 2, C = 6; median 2.
    model = _ec_only(np.full(len(ENZYMES), np.nan), ROWS)
    ids, values, result = level_complex_abundances(model, ["A", "B", "C"], [1.0, 4.0, 6.0])
    levelled = dict(zip(ids, values))
    assert levelled == pytest.approx({"A": 2.0, "B": 4.0, "C": 2.0})
    assert sorted(result.adjusted) == ["A", "C"]


def test_mean_is_available():
    model = _ec_only(np.full(len(ENZYMES), np.nan), ROWS)
    ids, values, _ = level_complex_abundances(model, ["A", "B", "C"], [1.0, 4.0, 6.0], statistic="mean")
    assert dict(zip(ids, values))["A"] == pytest.approx(3.0)


def test_shared_subunits_stay_as_measured_and_missing_ones_are_filled():
    model = _ec_only(np.full(len(ENZYMES), np.nan), ROWS)
    ids, values, result = level_complex_abundances(model, ["D", "S"], [1.0, 9.0])
    levelled = dict(zip(ids, values))
    assert levelled["S"] == 9.0             # shared: untouched and not in the summary
    assert levelled["D"] == levelled["F"] == 1.0
    assert "F" in result.added


def test_filling_can_be_switched_off():
    model = _ec_only(np.full(len(ENZYMES), np.nan), ROWS)
    ids, _, _ = level_complex_abundances(model, ["D"], [1.0], fill_missing=False)
    assert "F" not in ids


def test_level_complex_concs_works_in_mass_units():
    concs = np.full(len(ENZYMES), np.nan)
    concs[ENZYMES.index("A")] = 100.0   # 1 mmol per copy at MW 100
    concs[ENZYMES.index("B")] = 1600.0  # 8 mmol, 4 per copy at MW 200
    concs[ENZYMES.index("C")] = 100.0   # 2 mmol per copy at MW 50
    model = _ec_only(concs, ROWS)
    level_complex_concs(model)
    new = dict(zip(ENZYMES, model.ec.concs))
    # median per copy is 2 mmol: A = 200 mg, B = 2 copies * 2 * 200 = 800 mg, C = 2 * 50 = 100 mg.
    assert new["A"] == pytest.approx(200.0)
    assert new["B"] == pytest.approx(800.0)
    assert new["C"] == pytest.approx(100.0)
    assert np.isnan(new["H"])


def test_an_unknown_statistic_is_rejected():
    model = _ec_only(np.full(len(ENZYMES), np.nan), ROWS)
    with pytest.raises(ValueError, match="statistic"):
        level_complex_abundances(model, ["A"], [1.0], statistic="max")


# --------------------------------------------------------------------------- #
# flexibilize_enz_concs(per_complex=True)
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path, gr_exp: float) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\norg_name = "test"\ngr_exp = {gr_exp}\nbio_rxn = "biomass"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _complex_toy(adapter):
    """A -> B catalysed by a two-subunit complex X1 + X2; growth equals the flux."""
    model = EcModel("toy", adapter=adapter)
    a_e, a_c, b_c = (cobra.Metabolite(m, compartment="c") for m in ("A_e", "A_c", "B_c"))
    pool = cobra.Metabolite("prot_pool", compartment="c")
    x1, x2 = cobra.Metabolite("prot_X1", compartment="c"), cobra.Metabolite("prot_X2", compartment="c")
    model.add_metabolites([a_e, a_c, b_c, pool, x1, x2])
    reactions = {
        "EX_A": ({a_e: -1}, (-1000, 0)),
        "TR_A": ({a_e: -1, a_c: 1}, (0, 1000)),
        "R_AB": ({a_c: -1, b_c: 1, x1: -1, x2: -1}, (0, 1000)),
        "biomass": ({b_c: -1}, (0, 1000)),
        "prot_pool_exchange": ({pool: 1}, (0, 1000)),
        "usage_prot_X1": ({pool: -1, x1: 1}, (0, 1.0)),
        "usage_prot_X2": ({pool: -1, x2: 1}, (0, 3.0)),
    }
    for rid, (mets, bounds) in reactions.items():
        r = cobra.Reaction(rid)
        r.add_metabolites(mets)
        r.bounds = bounds
        model.add_reactions([r])
    model.objective = "biomass"
    model.ec = EcData(
        rxns=["R_AB"], kcat=np.array([1.0]), source=[""], notes=[""], eccodes=[""],
        genes=["g_X1", "g_X2"], enzymes=["X1", "X2"], mw=np.array([100.0, 100.0]),
        sequence=["", ""], concs=np.array([1.0, 3.0]),
        rxn_enz_mat=sparse.csr_matrix([[1.0, 1.0]]),
    )
    return model


def test_without_per_complex_only_the_limiting_subunit_is_raised(tmp_path):
    model = _complex_toy(_adapter(tmp_path, gr_exp=2.0))
    result = flexibilize_enz_concs(model, verbose=False)
    assert result.uniprot_ids == ["X1"]
    assert model.reactions.usage_prot_X2.upper_bound == pytest.approx(3.0)


def test_per_complex_raises_every_subunit_by_the_same_factor(tmp_path):
    model = _complex_toy(_adapter(tmp_path, gr_exp=2.0))
    result = flexibilize_enz_concs(model, verbose=False, per_complex=True)
    assert sorted(result.uniprot_ids) == ["X1", "X2"]
    factor_1 = model.reactions.usage_prot_X1.upper_bound / 1.0
    factor_2 = model.reactions.usage_prot_X2.upper_bound / 3.0
    assert factor_1 == pytest.approx(factor_2)
    assert model.optimize().objective_value == pytest.approx(2.0, rel=5e-3)
