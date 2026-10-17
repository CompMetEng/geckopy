"""Cross-implementation parity: geckopy vs the MATLAB GECKO Toolbox.

This module is a 1:1 port of GECKO's own unit-test suite,
``test/unit_tests/geckoCoreFunctionTests.m`` (plus its
``getBaseExpTstEcModelProperties.m`` helper), run against geckopy on the
same ``ecTestGEM`` fixture that MATLAB uses.

Why this file exists
--------------------
Every other test in this suite asserts what *geckopy* should do. This one
asserts what *MATLAB GECKO* does: every expected value below is copied
verbatim out of the MATLAB sources, not re-derived from geckopy's output.
That makes it the executable definition of "geckopy builds the same
ecModel as the GECKO Toolbox" -- if the two implementations drift, this
file fails and nothing else has to.

Each test carries the name of the MATLAB test case it mirrors
(``tc0001`` ... ``tc0013``), so a failure points straight at the MATLAB
function to compare against.

Scope and limits
----------------
- The fixture is GECKO's own ``ecTestGEM``: 5 genes, 7 reactions, a
  hand-written BRENDA/UniProt/KEGG/ComplexPortal/proteomics snapshot.
  Parity on ecTestGEM does not prove parity on yeast-GEM; it proves the
  core functions agree on every behaviour MATLAB itself pins down.
  Whole-GEM parity needs MATLAB in the loop (see
  ``CompMetEng/raven-gecko-parity``).
- Stages that need the network (UniProt/KEGG/ComplexPortal downloads,
  DLKcat inference) are out of scope here, exactly as in MATLAB's suite,
  which also runs them off stored files.
- The geckopy fixture files under ``examples/ecTestGEM/`` are identical to
  GECKO's ``test/unit_tests/ecTestGEM/`` (modulo line endings and the
  BRENDA text files, which geckopy stores as the ``kcat.tsv`` / ``mw.tsv`` /
  ``sa.tsv`` TSVs its loader reads).

MATLAB reference: GECKO commit c9101f17.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest

from geckopy import (
    EcModel,
    ModelAdapter,
    apply_complex_data,
    apply_custom_kcats,
    apply_kcat_constraints,
    apply_kcat_list,
    constrain_enz_concs,
    fill_eccodes_from_database,
    fill_eccodes_from_gem,
    fill_enz_concs,
    fill_kcats_from_isozymes,
    find_met_smiles,
    flexibilize_enz_concs,
    fuzzy_kcat_matching,
    load_brenda_data,
    load_conventional_gem,
    load_dlkcat_ignore_lists,
    load_ec_model,
    load_phyl_dist,
    load_prot_data,
    load_uniprot_tsv,
    make_ec_model,
    merge_kcats,
    save_ec_model,
    set_prot_pool_size,
    write_dlkcat_input,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
DATA_DIR = EXAMPLE_DIR / "data"


# --------------------------------------------------------------------------- #
# Fixture construction (mirrors getGeckoTestModel.m + TestGEMAdapter.m)
# --------------------------------------------------------------------------- #

def _test_adapter() -> ModelAdapter:
    """Load the ecTestGEM adapter subclass (port of TestGEMAdapter.m)."""
    spec = importlib.util.spec_from_file_location(
        "ectestgem_adapter_parity", EXAMPLE_DIR / "adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TestGEMAdapter.from_folder(EXAMPLE_DIR)


_GEM_CACHE: cobra.Model | None = None


def _test_gem() -> cobra.Model:
    """The conventional test GEM (port of getGeckoTestModel.m).

    The SBML parse is cached; every caller gets an independent copy.
    """
    global _GEM_CACHE
    if _GEM_CACHE is None:
        _GEM_CACHE = cobra.io.read_sbml_model(str(_test_adapter().params.conv_gem))
    return _GEM_CACHE.copy()


def _ec_model(gecko_light: bool = False) -> EcModel:
    """``makeEcModel(getGeckoTestModel(), gecko_light, adapter)``.

    Rebuilt per call (~0.04 s) rather than cached and copied -- the build
    is cheap enough that no cache is worth the risk of tests leaking
    mutations into each other.
    """
    return make_ec_model(_test_gem(), _test_adapter(), gecko_light=gecko_light)


@pytest.fixture
def adapter() -> ModelAdapter:
    return _test_adapter()


@pytest.fixture
def ec_model() -> EcModel:
    return _ec_model(False)


@pytest.fixture
def light_ec_model() -> EcModel:
    return _ec_model(True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rxn_ids(model: cobra.Model) -> list[str]:
    return [r.id for r in model.reactions]


def _met_names(model: cobra.Model) -> list[str]:
    return [m.name for m in model.metabolites]


def _stoichiometry(model: cobra.Model) -> dict[tuple[str, str], float]:
    """Non-zero S entries keyed by ``(metabolite name, reaction id)``.

    MATLAB compares a dense ``S`` whose row/column order is asserted
    separately; keying by name/id makes the stoichiometry check
    independent of that ordering, so an ordering regression fails one
    test rather than every test.
    """
    return {
        (met.name, rxn.id): coef
        for rxn in model.reactions
        for met, coef in rxn.metabolites.items()
        if coef != 0
    }


def _dense(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _s_coef(model: cobra.Model, met_id: str, rxn_id: str) -> float:
    rxn = model.reactions.get_by_id(rxn_id)
    for met, coef in rxn.metabolites.items():
        if met.id == met_id:
            return coef
    return 0.0


def _prot_costs(model: cobra.Model, rxn_id: str) -> list[float]:
    """The five ``prot_P*`` coefficients of one reaction, MATLAB's row order."""
    return [_s_coef(model, f"prot_P{i}", rxn_id) for i in range(1, 6)]


# --------------------------------------------------------------------------- #
# Expected values, copied from getBaseExpTstEcModelProperties.m
# --------------------------------------------------------------------------- #

EXP_RXNS = [
    "R1", "R1_REV", "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2",
    "R3", "R4", "R5", "S1", "S2",
    "usage_prot_P1", "usage_prot_P2", "usage_prot_P3", "usage_prot_P4",
    "usage_prot_P5", "prot_pool_exchange",
]

EXP_MET_NAMES = [
    "e1", "e2", "m1", "m2",
    "prot_P1", "prot_P2", "prot_P3", "prot_P4", "prot_P5", "prot_pool",
]

# expS from getBaseExpTstEcModelProperties.m, transcribed as
# (metabolite name, reaction id) -> coefficient.
EXP_S = {
    ("e1", "R1"): -1, ("m1", "R1"): 1,
    ("e1", "R1_REV"): 1, ("m1", "R1_REV"): -1,
    ("m1", "R2_EXP_1"): -1, ("m2", "R2_EXP_1"): 1,
    ("m1", "R2_EXP_2"): -1, ("m2", "R2_EXP_2"): 1,
    ("m1", "R2_REV_EXP_1"): 1, ("m2", "R2_REV_EXP_1"): -1,
    ("m1", "R2_REV_EXP_2"): 1, ("m2", "R2_REV_EXP_2"): -1,
    ("m1", "R3"): -1, ("m2", "R3"): 1,
    ("m1", "R4"): -1, ("m2", "R4"): 1,
    ("m2", "R5"): -1, ("e2", "R5"): 1,
    ("e1", "S1"): -1,
    ("e2", "S2"): -1,
    # Protein reactions, forward direction: prot_pool -> prot_<id>
    ("prot_pool", "usage_prot_P1"): -1, ("prot_P1", "usage_prot_P1"): 1,
    ("prot_pool", "usage_prot_P2"): -1, ("prot_P2", "usage_prot_P2"): 1,
    ("prot_pool", "usage_prot_P3"): -1, ("prot_P3", "usage_prot_P3"): 1,
    ("prot_pool", "usage_prot_P4"): -1, ("prot_P4", "usage_prot_P4"): 1,
    ("prot_pool", "usage_prot_P5"): -1, ("prot_P5", "usage_prot_P5"): 1,
    ("prot_pool", "prot_pool_exchange"): 1,
}

EXP_EC_RXNS_FULL = [
    "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2", "R3", "R5",
]
EXP_EC_RXNS_LIGHT = [
    "001_R2", "002_R2", "001_R3", "001_R5", "001_R2_REV", "002_R2_REV",
]
EXP_EC_GENES = ["G1", "G2", "G3", "G4", "G5"]
EXP_ENZYMES = ["P1", "P2", "P3", "P4", "P5"]
EXP_MW = [10000.0, 20000.0, 30000.0, 40000.0, 50000.0]
EXP_SEQUENCE = ["MRAL", "MNTD", "MSYN", "MDFM", "MLFK"]


def _exp_rxn_enz_mat(pairs: dict[int, dict[int, float]], n_rxns: int = 6) -> np.ndarray:
    """Build a rxnEnzMat from MATLAB's 1-based ``expRxnEnzMat(i,j) = v``."""
    mat = np.zeros((n_rxns, len(EXP_EC_GENES)))
    for i, cols in pairs.items():
        for j, val in cols.items():
            mat[i - 1, j - 1] = val
    return mat


# ========================================================================== #
# tc0001 -- testmakeEcModelFullModel
# ========================================================================== #

def test_tc0001_full_model_reaction_order(ec_model):
    """MATLAB sorts identifiers inside makeEcModel (sortIdentifiers, full
    models only), so the metabolic block comes out alphabetically and the
    protein block is appended after it."""
    assert _rxn_ids(ec_model) == EXP_RXNS


def test_tc0001_full_model_metabolite_order(ec_model):
    assert _met_names(ec_model) == EXP_MET_NAMES


def test_tc0001_full_model_stoichiometry(ec_model):
    assert _stoichiometry(ec_model) == EXP_S


def test_tc0001_full_model_ec_rxns(ec_model):
    assert ec_model.ec.rxns == EXP_EC_RXNS_FULL


def test_tc0001_full_model_ec_array_lengths(ec_model):
    n = len(ec_model.ec.rxns)
    assert len(ec_model.ec.kcat) == n
    assert len(ec_model.ec.source) == n
    assert len(ec_model.ec.notes) == n
    assert len(ec_model.ec.eccodes) == n


def test_tc0001_full_model_enzyme_data(ec_model):
    assert ec_model.ec.genes == EXP_EC_GENES
    assert ec_model.ec.enzymes == EXP_ENZYMES
    assert list(ec_model.ec.sequence) == EXP_SEQUENCE
    np.testing.assert_array_equal(np.asarray(ec_model.ec.mw, dtype=float), EXP_MW)
    assert len(ec_model.ec.concs) == len(EXP_EC_GENES)


def test_tc0001_full_model_rxn_enz_mat(ec_model):
    expected = _exp_rxn_enz_mat({
        1: {1: 1, 2: 1},
        2: {3: 1},
        3: {1: 1, 2: 1},
        4: {3: 1},
        5: {4: 1},
        6: {5: 1},
    })
    np.testing.assert_array_equal(_dense(ec_model.ec.rxn_enz_mat), expected)


# ========================================================================== #
# tc0002 -- testmakeEcModelLightModel
# ========================================================================== #

def test_tc0002_light_model_reaction_order(light_ec_model):
    """``expRxns = [model.rxns; 'R1_REV'; 'R2_REV'; 'prot_pool_exchange']``
    -- light models are not sorted in MATLAB, so the original GEM order is
    kept and the new reactions are appended."""
    base = _rxn_ids(_test_gem())
    assert _rxn_ids(light_ec_model) == base + [
        "R1_REV", "R2_REV", "prot_pool_exchange",
    ]


def test_tc0002_light_model_metabolite_order(light_ec_model):
    assert _met_names(light_ec_model) == _met_names(_test_gem()) + ["prot_pool"]


def test_tc0002_light_model_stoichiometry(light_ec_model):
    """``expS = [model.S, -model.S(:,2:3), 0; 0 ... 1]`` -- columns 2:3 of
    the base model are R1 and R2, negated to give R1_REV and R2_REV; the
    prot_pool row carries a single +1 in prot_pool_exchange."""
    gem = _test_gem()
    expected = _stoichiometry(gem)
    for rxn_id in ("R1", "R2"):
        for met, coef in gem.reactions.get_by_id(rxn_id).metabolites.items():
            expected[(met.name, f"{rxn_id}_REV")] = -coef
    expected[("prot_pool", "prot_pool_exchange")] = 1
    assert _stoichiometry(light_ec_model) == expected


def test_tc0002_light_model_ec_rxns(light_ec_model):
    assert light_ec_model.ec.rxns == EXP_EC_RXNS_LIGHT


def test_tc0002_light_model_ec_array_lengths(light_ec_model):
    n = len(light_ec_model.ec.rxns)
    assert len(light_ec_model.ec.kcat) == n
    assert len(light_ec_model.ec.source) == n
    assert len(light_ec_model.ec.notes) == n
    assert len(light_ec_model.ec.eccodes) == n


def test_tc0002_light_model_enzyme_data(light_ec_model):
    assert light_ec_model.ec.genes == EXP_EC_GENES
    assert light_ec_model.ec.enzymes == EXP_ENZYMES
    assert list(light_ec_model.ec.sequence) == EXP_SEQUENCE
    np.testing.assert_array_equal(
        np.asarray(light_ec_model.ec.mw, dtype=float), EXP_MW
    )
    assert len(light_ec_model.ec.concs) == len(EXP_EC_GENES)


def test_tc0002_light_model_rxn_enz_mat(light_ec_model):
    expected = _exp_rxn_enz_mat({
        1: {1: 1, 2: 1},
        2: {3: 1},
        3: {4: 1},
        4: {5: 1},
        5: {1: 1, 2: 1},
        6: {3: 1},
    })
    np.testing.assert_array_equal(_dense(light_ec_model.ec.rxn_enz_mat), expected)


# ========================================================================== #
# tc0003 / tc0004 -- testapplyComplexData (full / light)
# ========================================================================== #

def test_tc0003_apply_complex_data_full(ec_model):
    apply_complex_data(ec_model, path=DATA_DIR / "ComplexPortal.json", apply=False)
    expected = _exp_rxn_enz_mat({
        1: {1: 1, 2: 2},
        2: {3: 1},
        3: {1: 1, 2: 2},
        4: {3: 1},
        5: {4: 1},
        6: {5: 1},
    })
    np.testing.assert_array_equal(_dense(ec_model.ec.rxn_enz_mat), expected)


def test_tc0004_apply_complex_data_light(light_ec_model):
    apply_complex_data(
        light_ec_model, path=DATA_DIR / "ComplexPortal.json", apply=False
    )
    expected = _exp_rxn_enz_mat({
        1: {1: 1, 2: 2},
        2: {3: 1},
        3: {4: 1},
        4: {5: 1},
        5: {1: 1, 2: 2},
        6: {3: 1},
    })
    np.testing.assert_array_equal(_dense(light_ec_model.ec.rxn_enz_mat), expected)


# ========================================================================== #
# tc0005 -- testsetProtPoolSize (full and light)
# ========================================================================== #

@pytest.mark.parametrize("gecko_light", [False, True], ids=["full", "light"])
def test_tc0005_set_prot_pool_size(gecko_light):
    """Sets the protein-pool exchange reaction's upper bound to
    p_tot*f*sigma*1000: adapter defaults give 0.5*4*0.5*1000 = 1000,
    and the explicit (p_tot=1, f=5, sigma=1) call gives 5000.
    """
    model = _ec_model(gecko_light)
    pool = model.reactions.get_by_id("prot_pool_exchange")
    set_prot_pool_size(model)
    assert pool.upper_bound == pytest.approx(1000)
    set_prot_pool_size(model, p_tot=1, f=5, sigma=1)
    assert pool.upper_bound == pytest.approx(5000)


# ========================================================================== #
# tc0006 -- testgetECfromGEM (full and light)
# ========================================================================== #

def test_tc0006_ec_from_gem_full(ec_model):
    fill_eccodes_from_gem(ec_model)
    assert list(ec_model.ec.eccodes) == [
        "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3",
    ]


def test_tc0006_ec_from_gem_full_subset(ec_model):
    fill_eccodes_from_gem(ec_model, ec_rxns=["R2_EXP_1", "R3"])
    assert list(ec_model.ec.eccodes) == ["1.1.1.1", "", "", "", "1.1.2.1", ""]


def test_tc0006_ec_from_gem_light(light_ec_model):
    fill_eccodes_from_gem(light_ec_model)
    assert list(light_ec_model.ec.eccodes) == [
        "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3", "1.1.1.1", "1.1.1.1",
    ]


def test_tc0006_ec_from_gem_light_subset(light_ec_model):
    fill_eccodes_from_gem(light_ec_model, ec_rxns=["001_R2", "001_R3"])
    assert list(light_ec_model.ec.eccodes) == [
        "1.1.1.1", "", "1.1.2.1", "", "", "",
    ]


# ========================================================================== #
# tc0007 -- testgetECfromDatabase (full and light)
# ========================================================================== #

def _uniprot_db():
    return load_uniprot_tsv(DATA_DIR / "uniprot.tsv")


def _kegg_db():
    from geckopy.databases.kegg_loader import load_kegg_tsv

    return load_kegg_tsv(DATA_DIR / "kegg.tsv")


def test_tc0007_ec_from_database_full(ec_model):
    """P5 has no EC number in uniprot.tsv, so MATLAB falls through to the
    KEGG table, which gives R5 its 1.1.1.3."""
    fill_eccodes_from_database(ec_model, _uniprot_db(), kegg_db=_kegg_db())
    assert list(ec_model.ec.eccodes) == [
        "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3",
    ]


def test_tc0007_ec_from_database_full_subset(ec_model):
    fill_eccodes_from_database(
        ec_model, _uniprot_db(), kegg_db=_kegg_db(), ec_rxns=["R2_EXP_1", "R3"]
    )
    assert list(ec_model.ec.eccodes) == ["1.1.1.1", "", "", "", "1.1.2.1", ""]


def test_tc0007_ec_from_database_light(light_ec_model):
    fill_eccodes_from_database(light_ec_model, _uniprot_db(), kegg_db=_kegg_db())
    assert list(light_ec_model.ec.eccodes) == [
        "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3", "1.1.1.1", "1.1.1.1",
    ]


def test_tc0007_ec_from_database_light_subset(light_ec_model):
    fill_eccodes_from_database(
        light_ec_model, _uniprot_db(), kegg_db=_kegg_db(),
        ec_rxns=["001_R2", "001_R3"],
    )
    assert list(light_ec_model.ec.eccodes) == [
        "1.1.1.1", "", "1.1.2.1", "", "", "",
    ]


# ========================================================================== #
# tc0008 -- testModelAdapterManager
# ========================================================================== #

def test_tc0008_adapter_loads(adapter):
    """MATLAB's ModelAdapterManager (a global default-adapter registry) has
    no geckopy counterpart: geckopy passes the adapter explicitly and
    attaches it to the model. The portable part of the MATLAB test is that
    loading the ecTestGEM adapter yields a populated adapter."""
    assert adapter is not None
    assert adapter.params.org_name == "testus testus"
    assert _ec_model(False).adapter is not None


# ========================================================================== #
# tc0009 -- testsaveECModel (save/load round trip + loadConventionalGEM)
# ========================================================================== #

def test_tc0009_save_load_round_trip(ec_model, tmp_path):
    path = tmp_path / "ecModel.yml"
    save_ec_model(ec_model, path)
    loaded = load_ec_model(path)

    assert _rxn_ids(loaded) == _rxn_ids(ec_model)
    assert _met_names(loaded) == _met_names(ec_model)
    assert _stoichiometry(loaded) == _stoichiometry(ec_model)
    assert [r.bounds for r in loaded.reactions] == [
        r.bounds for r in ec_model.reactions
    ]
    assert loaded.ec.rxns == ec_model.ec.rxns
    assert loaded.ec.genes == ec_model.ec.genes
    assert loaded.ec.enzymes == ec_model.ec.enzymes
    assert list(loaded.ec.sequence) == list(ec_model.ec.sequence)
    np.testing.assert_array_equal(
        np.asarray(loaded.ec.mw, dtype=float),
        np.asarray(ec_model.ec.mw, dtype=float),
    )
    np.testing.assert_array_equal(
        _dense(loaded.ec.rxn_enz_mat), _dense(ec_model.ec.rxn_enz_mat)
    )


def test_tc0009_load_conventional_gem(adapter):
    model = load_conventional_gem(adapter)
    assert _rxn_ids(model) == _rxn_ids(_test_gem())
    assert _stoichiometry(model) == _stoichiometry(_test_gem())


# ========================================================================== #
# tc0010 -- testfuzzyKcatMatching (full and light, all rxns and a subset)
# ========================================================================== #

def _brenda():
    return load_brenda_data(DATA_DIR)


def _phyl_dist():
    return load_phyl_dist(DATA_DIR / "PhylDist.mat")


def _fuzzy(model, ec_rxns=None) -> pd.DataFrame:
    fill_eccodes_from_gem(model)
    return fuzzy_kcat_matching(model, _brenda(), _phyl_dist(), ec_rxns=ec_rxns)


def test_tc0010_fuzzy_full_all_rxns(ec_model):
    """MATLAB comment: 'substrate is more important than organism, and
    wildcard comes last'."""
    df = _fuzzy(ec_model)
    assert list(df["rxn_id"]) == EXP_EC_RXNS_FULL
    assert list(df["kcat"]) == [1, 1, 10, 10, 100, 1]
    assert [list(s) for s in df["substrates"]] == [
        ["m1"], ["m1"], ["m2"], ["m2"], ["m1"], ["m2"],
    ]
    assert list(df["eccode"]) == [
        "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3",
    ]
    assert list(df["wildcard_level"]) == [0, 0, 0, 0, 1, 1]
    assert list(df["origin"]) == [1, 1, 2, 2, 3, 3]


def test_tc0010_fuzzy_full_subset(ec_model):
    df = _fuzzy(ec_model, ec_rxns=["R2_REV_EXP_1"])
    assert list(df["rxn_id"]) == ["R2_REV_EXP_1"]
    assert list(df["kcat"]) == [10]
    assert [list(s) for s in df["substrates"]] == [["m2"]]
    assert list(df["eccode"]) == ["1.1.1.1"]
    assert list(df["wildcard_level"]) == [0]
    assert list(df["origin"]) == [2]


def test_tc0010_fuzzy_light_all_rxns(light_ec_model):
    df = _fuzzy(light_ec_model)
    assert list(df["rxn_id"]) == EXP_EC_RXNS_LIGHT
    assert list(df["kcat"]) == [1, 1, 100, 1, 10, 10]
    assert [list(s) for s in df["substrates"]] == [
        ["m1"], ["m1"], ["m1"], ["m2"], ["m2"], ["m2"],
    ]
    assert list(df["eccode"]) == [
        "1.1.1.1", "1.1.1.1", "1.1.2.1", "1.1.1.3", "1.1.1.1", "1.1.1.1",
    ]
    assert list(df["wildcard_level"]) == [0, 0, 1, 1, 0, 0]
    assert list(df["origin"]) == [1, 1, 3, 3, 2, 2]


def test_tc0010_fuzzy_light_subset(light_ec_model):
    df = _fuzzy(light_ec_model, ec_rxns=["001_R2_REV"])
    assert list(df["rxn_id"]) == ["001_R2_REV"]
    assert list(df["kcat"]) == [10]
    assert [list(s) for s in df["substrates"]] == [["m2"]]
    assert list(df["eccode"]) == ["1.1.1.1"]
    assert list(df["wildcard_level"]) == [0]
    assert list(df["origin"]) == [2]


# ========================================================================== #
# tc0011 -- testKcats
#
# Covers writeDLKcatInput, mergeDLKcatAndFuzzyKcats, selectKcatValue,
# applyKcatConstraints, getKcatAcrossIsozymes and applyCustomKcats, plus
# the full-vs-light growth-rate equivalence.
# ========================================================================== #

def _extended_gem() -> cobra.Model:
    """getGeckoTestModel() + the two extra reactions tc0011 adds.

    R2a duplicates R2's GPR but carries no EC code (so DLKcat has to fill
    it); R3b duplicates R3 including its EC code but is absent from the
    DLKcat list (so the fuzzy wildcard match has to fill it).
    """
    model = _test_gem()
    m1 = model.metabolites.get_by_id("m1c")
    m2 = model.metabolites.get_by_id("m2c")
    for rxn_id, gpr, eccode in (
        ("R2a", "(G1 and G2) or G3", None),
        ("R3b", "G4", "1.1.2.1"),
    ):
        rxn = cobra.Reaction(rxn_id)
        rxn.bounds = (0.0, 1000.0)
        rxn.add_metabolites({m1: -1, m2: 1})
        model.add_reactions([rxn])
        added = model.reactions.get_by_id(rxn_id)
        added.gene_reaction_rule = gpr
        if eccode:
            added.annotation["ec-code"] = eccode
    return model


# ec.rxns of the extended full model, in MATLAB's (sorted) order.
EXP_EXT_EC_RXNS = [
    "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2",
    "R2a_EXP_1", "R2a_EXP_2", "R3", "R3b", "R5",
]

def _extended_ec_model(gecko_light: bool = False) -> EcModel:
    """Extended model taken through makeEcModel + getECfromGEM + complexes."""
    model = make_ec_model(_extended_gem(), _test_adapter(), gecko_light=gecko_light)
    fill_eccodes_from_gem(model)
    apply_complex_data(model, path=DATA_DIR / "ComplexPortal.json", apply=False)
    return model


def _dlkcat_list(rxns, genes, substrates, kcats) -> pd.DataFrame:
    """A DLKcat-sourced kcat list in geckopy's canonical schema."""
    return pd.DataFrame({
        "rxn_id": rxns,
        "source": ["DLKcat"] * len(rxns),
        "eccode": [""] * len(rxns),
        "substrates": [[s] for s in substrates],
        "genes": [[g] for g in genes],
        "kcat": [float(k) for k in kcats],
        "wildcard_level": pd.array([pd.NA] * len(rxns), dtype="Int64"),
        "origin": pd.array([pd.NA] * len(rxns), dtype="Int64"),
    })


DLKCAT_FULL = dict(
    rxns=[
        "R2_EXP_1", "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_1",
        "R2_REV_EXP_2", "R2a_EXP_1", "R2a_EXP_1", "R2a_EXP_2", "R3", "R5",
    ],
    genes=["G1", "G2", "G3", "G1", "G2", "G3", "G1", "G2", "G3", "G4", "G5"],
    substrates=["m1", "m1", "m1", "m2", "m2", "m2", "m1", "m1", "m1", "m1", "m2"],
    kcats=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011],
)

DLKCAT_LIGHT = dict(
    rxns=[
        "001_R2", "001_R2", "002_R2", "001_R3", "001_R5", "001_R2a",
        "001_R2a", "002_R2a", "001_R2_REV", "001_R2_REV", "002_R2_REV",
    ],
    genes=["G1", "G2", "G3", "G4", "G5", "G1", "G2", "G3", "G1", "G2", "G3"],
    substrates=["m1", "m1", "m1", "m1", "m2", "m1", "m1", "m1", "m2", "m2", "m2"],
    kcats=[1001, 1002, 1003, 1010, 1011, 1007, 1008, 1009, 1004, 1005, 1006],
)

# mergeDLKcatAndFuzzyKcats(dlkcatList, kcatListFuzzy, 6, 6, 1)
#
# MATLAB's binary merge always emits "fuzzy followed by dlkcat" regardless
# of argument order. geckopy's merge_kcats is n-ary and concatenates the
# surviving rows list by list, so the equivalent call passes the fuzzy list
# first: merge_kcats(fuzzy, dlkcat, ...). Note that the deprecated
# merge_dlkcat_and_fuzzy_kcats alias is merge_kcats, so it inherits this
# convention rather than MATLAB's (dlkcat, fuzzy) signature.
_MERGE_KWARGS = dict(
    source_priority=("database_top", "dlkcat", "database_bottom"),
    top_origin_limit=6,
    bottom_origin_limit=6,
    wildcard_limit=1,
)


def test_tc0011_write_dlkcat_input(tmp_path):
    """MATLAB checks rows 1, 2, 3 and 5 of the written table (reaction,
    gene, substrate, sequence); rows 4 and 6 (SMILES, kcat placeholder)
    are skipped there and here."""
    model = _extended_ec_model()
    table = write_dlkcat_input(
        model,
        tmp_path / "DLKcat_input_test.tsv",
        load_dlkcat_ignore_lists(),
        only_with_smiles=False,
        overwrite=True,
    )
    assert (tmp_path / "DLKcat_input_test.tsv").is_file()
    assert list(table["rxn_id"]) == [
        "R2_EXP_1", "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_1",
        "R2_REV_EXP_2", "R2a_EXP_1", "R2a_EXP_1", "R2a_EXP_2", "R3", "R3b", "R5",
    ]
    assert list(table["gene"]) == [
        "G1", "G2", "G3", "G1", "G2", "G3", "G1", "G2", "G3", "G4", "G4", "G5",
    ]
    assert list(table["substrate"]) == [
        "m1", "m1", "m1", "m2", "m2", "m2", "m1", "m1", "m1", "m1", "m1", "m2",
    ]
    assert list(table["sequence"]) == [
        "MRAL", "MNTD", "MSYN", "MRAL", "MNTD", "MSYN",
        "MRAL", "MNTD", "MSYN", "MDFM", "MDFM", "MLFK",
    ]


def test_tc0011_merge_dlkcat_and_fuzzy():
    """MATLAB's expectations, verbatim: R2* keep their (good) BRENDA match,
    R3b keeps its wildcard BRENDA match because DLKcat never predicted it,
    R2a* have no EC code so DLKcat wins, R3's wildcard BRENDA match loses
    to DLKcat, and R5 has no BRENDA kcat at all."""
    model = _extended_ec_model()
    fuzzy = fuzzy_kcat_matching(model, _brenda(), _phyl_dist())
    merged = merge_kcats(fuzzy, _dlkcat_list(**DLKCAT_FULL), **_MERGE_KWARGS)

    assert [s.lower() for s in merged["source"]] == [
        "brenda", "brenda", "brenda", "brenda", "brenda",
        "dlkcat", "dlkcat", "dlkcat", "dlkcat", "dlkcat",
    ]
    assert list(merged["rxn_id"]) == [
        "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2", "R3b",
        "R2a_EXP_1", "R2a_EXP_1", "R2a_EXP_2", "R3", "R5",
    ]
    assert list(merged["kcat"]) == [1, 1, 10, 10, 100, 1007, 1008, 1009, 1010, 1011]
    assert list(merged["eccode"]) == [
        "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.2.1", "", "", "", "", "",
    ]


def _kcat_by_rxn(model: EcModel) -> dict[str, float]:
    return dict(zip(model.ec.rxns, [float(k) for k in model.ec.kcat]))


def test_tc0011_select_kcat_value():
    """The highest kcat wins per reaction: R2a_EXP_1 takes 1008, not 1007."""
    model = _extended_ec_model()
    fuzzy = fuzzy_kcat_matching(model, _brenda(), _phyl_dist())
    merged = merge_kcats(fuzzy, _dlkcat_list(**DLKCAT_FULL), **_MERGE_KWARGS)
    apply_kcat_list(model, merged, criteria="max")

    assert _kcat_by_rxn(model) == dict(zip(
        EXP_EXT_EC_RXNS, [1, 1, 10, 10, 1008, 1009, 1010, 100, 1011]
    ))


def _constrained_extended_model() -> EcModel:
    model = _extended_ec_model()
    fuzzy = fuzzy_kcat_matching(model, _brenda(), _phyl_dist())
    merged = merge_kcats(fuzzy, _dlkcat_list(**DLKCAT_FULL), **_MERGE_KWARGS)
    apply_kcat_list(model, merged, criteria="max")
    return model


def test_tc0011_apply_kcat_constraints_subset():
    """Constraining only {R3, R5} must leave every other reaction's protein
    cost at zero."""
    model = _constrained_extended_model()
    apply_kcat_constraints(model, ["R3", "R5"])

    for rxn_id in (
        "R1", "R1_REV", "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2",
        "R2a_EXP_1", "R2a_EXP_2", "R3b", "R4", "S1", "S2",
    ):
        assert _prot_costs(model, rxn_id) == pytest.approx([0, 0, 0, 0, 0], abs=1e-10)
    assert _prot_costs(model, "R3") == pytest.approx(
        [0, 0, 0, -40000 / 1010 / 3600, 0], abs=1e-10
    )
    assert _prot_costs(model, "R5") == pytest.approx(
        [0, 0, 0, 0, -50000 / 1011 / 3600], abs=1e-10
    )


def test_tc0011_apply_kcat_constraints_all():
    model = _constrained_extended_model()
    apply_kcat_constraints(model)

    expected = {
        "R1": [0, 0, 0, 0, 0],
        "R1_REV": [0, 0, 0, 0, 0],
        # MW 10000 + 2*20000 (P1 + 2*P2), kcat 1
        "R2_EXP_1": [-10000 / 1 / 3600, -(2 * 20000) / 1 / 3600, 0, 0, 0],
        "R2_EXP_2": [0, 0, -30000 / 1 / 3600, 0, 0],
        "R2_REV_EXP_1": [-10000 / 10 / 3600, -(2 * 20000) / 10 / 3600, 0, 0, 0],
        "R2_REV_EXP_2": [0, 0, -30000 / 10 / 3600, 0, 0],
        # no EC code, so kcat from DLKcat: max(1007, 1008) = 1008
        "R2a_EXP_1": [-10000 / 1008 / 3600, -(2 * 20000) / 1008 / 3600, 0, 0, 0],
        "R2a_EXP_2": [0, 0, -30000 / 1009 / 3600, 0, 0],
        "R3": [0, 0, 0, -40000 / 1010 / 3600, 0],
        "R3b": [0, 0, 0, -40000 / 100 / 3600, 0],
        "R4": [0, 0, 0, 0, 0],
        "R5": [0, 0, 0, 0, -50000 / 1011 / 3600],
        "S1": [0, 0, 0, 0, 0],
        "S2": [0, 0, 0, 0, 0],
    }
    for rxn_id, costs in expected.items():
        assert _prot_costs(model, rxn_id) == pytest.approx(costs, abs=1e-10), rxn_id


def test_tc0011_get_kcat_across_isozymes():
    """Zeroing R2_EXP_2's kcat and refilling from its isozyme restores it."""
    model = _constrained_extended_model()
    apply_kcat_constraints(model)
    model.ec.kcat[model.ec.rxns.index("R2_EXP_2")] = 0
    fill_kcats_from_isozymes(model)

    assert _kcat_by_rxn(model) == dict(zip(
        EXP_EXT_EC_RXNS, [1, 1, 10, 10, 1008, 1009, 1010, 100, 1011]
    ))


def test_tc0011_apply_custom_kcats_from_file():
    """customKcats.tsv sets P3 -> 200, P1+P2 -> 100, and R2_REV/R5 -> 50."""
    model = _constrained_extended_model()
    apply_kcat_constraints(model)
    apply_custom_kcats(model, path=DATA_DIR / "customKcats.tsv", apply=False)

    assert _kcat_by_rxn(model) == dict(zip(
        EXP_EXT_EC_RXNS, [100, 200, 50, 50, 100, 200, 1010, 100, 50]
    ))


def _constrained_extended_light_model() -> EcModel:
    light = _extended_ec_model(gecko_light=True)
    fuzzy = fuzzy_kcat_matching(light, _brenda(), _phyl_dist())
    merged = merge_kcats(fuzzy, _dlkcat_list(**DLKCAT_LIGHT), **_MERGE_KWARGS)
    apply_kcat_list(light, merged, criteria="max")
    return light


def test_tc0011_apply_kcat_constraints_light():
    """Light models carry one prot_pool coefficient per reaction: the
    cheapest isozyme (min MW_sum/kcat) rather than one column per isozyme."""
    light = _constrained_extended_light_model()
    apply_kcat_constraints(light)

    expected = {
        "R1": 0,
        "R2": -min(10000 / 1 / 3600 + (2 * 20000) / 1 / 3600, 30000 / 1 / 3600),
        "R2_REV": -min(
            10000 / 10 / 3600 + (2 * 20000) / 10 / 3600, 30000 / 10 / 3600
        ),
        "R2a": -min(
            10000 / 1008 / 3600 + (2 * 20000) / 1008 / 3600, 30000 / 1009 / 3600
        ),
        "R3": -40000 / 1010 / 3600,
        "R3b": -40000 / 100 / 3600,
        "R4": 0,
        "R5": -50000 / 1011 / 3600,
        "S1": 0,
        "S2": 0,
    }
    for rxn_id, coef in expected.items():
        assert _s_coef(light, "prot_pool", rxn_id) == pytest.approx(
            coef, abs=1e-10
        ), rxn_id


def test_tc0011_full_and_light_have_the_same_growth_rate():
    """The MATLAB test's final check: the same fixture constrained as a
    full and as a light ecModel must optimise to the same objective."""
    full = _constrained_extended_model()
    apply_kcat_constraints(full)
    set_prot_pool_size(full)

    light = _constrained_extended_light_model()
    apply_kcat_constraints(light)
    set_prot_pool_size(light)

    assert full.slim_optimize() == pytest.approx(light.slim_optimize(), abs=1e-10)


# ========================================================================== #
# tc0012 -- testfindMetSmiles (from the stored smilesDB, no network)
# ========================================================================== #

def test_tc0012_find_met_smiles(ec_model):
    find_met_smiles(ec_model, cache_path=DATA_DIR / "smilesDB.tsv")
    assert [m.annotation.get("smiles", "") for m in ec_model.metabolites] == [
        "C(C1C)O", "C1C(=NC2)", "C(C1C)O", "C1C(=NC2)", "", "", "", "", "", "",
    ]


# ========================================================================== #
# tc0013 -- testProteomicsIntegration
# ========================================================================== #

EXP_ABUNDANCES = [0.7292388, 0.03692241, 0.318175, 5.1959184, 0.15647268]


def _proteomics_ready_model() -> EcModel:
    """makeEcModel -> getECfromGEM -> applyComplexData -> fuzzy -> select
    -> applyKcatConstraints -> setProtPoolSize, as tc0013 does."""
    model = _ec_model()
    fill_eccodes_from_gem(model)
    apply_complex_data(model, path=DATA_DIR / "ComplexPortal.json", apply=False)
    fuzzy = fuzzy_kcat_matching(model, _brenda(), _phyl_dist())
    apply_kcat_list(model, fuzzy, criteria="max")
    apply_kcat_constraints(model)
    set_prot_pool_size(model)
    return model


def test_tc0013_load_prot_data():
    prot_data = load_prot_data(DATA_DIR / "proteomics.tsv", [1])
    assert list(prot_data.uniprot_ids) == EXP_ENZYMES
    np.testing.assert_allclose(
        np.asarray(prot_data.abundances, dtype=float).ravel(), EXP_ABUNDANCES
    )


def test_tc0013_fill_enz_concs():
    model = _proteomics_ready_model()
    fill_enz_concs(model, load_prot_data(DATA_DIR / "proteomics.tsv", [1]))
    np.testing.assert_allclose(
        np.asarray(model.ec.concs, dtype=float), EXP_ABUNDANCES
    )


def test_tc0013_constrain_enz_concs():
    model = _proteomics_ready_model()
    fill_enz_concs(model, load_prot_data(DATA_DIR / "proteomics.tsv", [1]))
    constrain_enz_concs(model)
    bounds = [
        model.reactions.get_by_id(f"usage_prot_{e}").upper_bound
        for e in model.ec.enzymes
    ]
    np.testing.assert_allclose(bounds, np.asarray(model.ec.concs, dtype=float))


def test_tc0013_flexibilize_enz_concs():
    """After flexibilisation the usage bounds must equal the returned
    flexibilised concentrations."""
    model = _proteomics_ready_model()
    fill_enz_concs(model, load_prot_data(DATA_DIR / "proteomics.tsv", [1]))
    constrain_enz_concs(model)
    result = flexibilize_enz_concs(model, exp_growth=0.4, verbose=False)

    bounds = [
        model.reactions.get_by_id(f"usage_prot_{p}").upper_bound
        for p in result.uniprot_ids
    ]
    np.testing.assert_allclose(bounds, np.asarray(result.flex_concs, dtype=float))
