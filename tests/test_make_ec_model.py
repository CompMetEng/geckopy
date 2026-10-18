"""Tests for the top-level make_ec_model orchestrator."""
import logging
from pathlib import Path

import cobra
import numpy as np
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.databases import UniprotDB, load_uniprot_tsv

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


# --------------------------------------------------------------------------- #
# End-to-end on the ecTestGEM fixture
# --------------------------------------------------------------------------- #

def _load_fresh_ectestgem() -> tuple[cobra.Model, ModelAdapter]:
    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    return model, adapter


def test_make_ec_model_returns_ec_model_instance():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    assert isinstance(ec_model, EcModel)


def test_make_ec_model_no_unmatched_genes_in_ectestgem(caplog):
    """All 5 genes in ecTestGEM match UniProt, so no warning is logged."""
    model, adapter = _load_fresh_ectestgem()
    with caplog.at_level(logging.WARNING, logger="geckopy.ec_model.make_ec_model"):
        make_ec_model(model, adapter)
    assert "not found in UniProt" not in caplog.text


def test_make_ec_model_populates_ec_rxns():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    assert sorted(ec_model.ec.rxns) == sorted([
        "R2_EXP_1", "R2_EXP_2",
        "R2_REV_EXP_1", "R2_REV_EXP_2",
        "R3", "R5",
    ])


def test_make_ec_model_does_not_split_exchange_reactions():
    """S1 and S2 are reversible exchanges in ecTestGEM. MATLAB's
    makeEcModel.m excludes exchanges from convertToIrrev (it passes
    nonExchRxns), so they stay reversible here too -- only R1 and R2
    (both internal, catalyzed reactions) get split."""
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    rxn_ids = {r.id for r in ec_model.reactions}
    assert "S1_REV" not in rxn_ids
    assert "S2_REV" not in rxn_ids
    s1 = ec_model.reactions.get_by_id("S1")
    assert s1.lower_bound < 0


def test_make_ec_model_populates_ec_genes_alphabetically():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    assert ec_model.ec.genes == ["G1", "G2", "G3", "G4", "G5"]
    assert ec_model.ec.enzymes == ["P1", "P2", "P3", "P4", "P5"]
    np.testing.assert_array_almost_equal(
        ec_model.ec.mw, [10000.0, 20000.0, 30000.0, 40000.0, 50000.0]
    )


def test_make_ec_model_kcat_initialized_zero():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    assert (ec_model.ec.kcat == 0).all()


def test_make_ec_model_adds_prot_metabolites():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    met_ids = {m.id for m in ec_model.metabolites}
    assert "prot_pool" in met_ids
    for p in ["prot_P1", "prot_P2", "prot_P3", "prot_P4", "prot_P5"]:
        assert p in met_ids


def test_make_ec_model_enzyme_notes_saved_as_raven_met_notes(tmp_path):
    """The enzyme and pool pseudometabolites carry RAVEN's free-text metNotes,
    as GECKO's makeEcModel writes them, so that RAVEN can read the YAML."""
    from ruamel.yaml import YAML
    from geckopy.utilities import save_ec_model
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    save_ec_model(ec_model, tmp_path / "ec.yml")
    doc = YAML(typ="safe").load((tmp_path / "ec.yml").read_text())
    mets = {m["id"]: m for m in doc["metabolites"]}
    assert mets["prot_P1"]["notes"] == "Enzyme-usage pseudometabolite"
    assert mets["prot_pool"]["notes"] == "Enzyme-usage protein pool"


def test_make_ec_model_adds_usage_and_exchange_reactions():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    rxn_ids = {r.id for r in ec_model.reactions}
    assert "prot_pool_exchange" in rxn_ids
    for u in [f"usage_prot_P{i}" for i in range(1, 6)]:
        assert u in rxn_ids


def test_make_ec_model_coupling_matrix_shape_and_values():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    mat = ec_model.ec.rxn_enz_mat.toarray()
    assert mat.shape == (6, 5)
    assert set(np.unique(mat)).issubset({0.0, 1.0})
    assert int(mat.sum()) == 8


def test_make_ec_model_validates_ec_consistency():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    ec_model.validate_ec()


def test_make_ec_model_attaches_adapter():
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)
    assert ec_model.adapter is adapter


def test_make_ec_model_does_not_mutate_input():
    """The input GEM must be left untouched; make_ec_model works on a copy.

    The preprocessing/expansion stages add _REV and _EXP_ reactions and
    prot_ pseudometabolites, and strip pseudoreaction GPRs, so mutation of
    the input would be visible in the reaction/metabolite sets, bounds and
    gene rules.
    """
    model, adapter = _load_fresh_ectestgem()
    rxn_ids_before = {r.id for r in model.reactions}
    met_ids_before = {m.id for m in model.metabolites}
    gene_ids_before = {g.id for g in model.genes}
    bounds_before = {r.id: r.bounds for r in model.reactions}
    grrules_before = {r.id: r.gene_reaction_rule for r in model.reactions}

    make_ec_model(model, adapter)

    assert {r.id for r in model.reactions} == rxn_ids_before
    assert {m.id for m in model.metabolites} == met_ids_before
    assert {g.id for g in model.genes} == gene_ids_before
    assert {r.id: r.bounds for r in model.reactions} == bounds_before
    assert {r.id: r.gene_reaction_rule for r in model.reactions} == grrules_before


# --------------------------------------------------------------------------- #
# Branching and error handling
# --------------------------------------------------------------------------- #

def test_make_ec_model_accepts_preloaded_uniprot_db():
    model, adapter = _load_fresh_ectestgem()
    db = load_uniprot_tsv(EXAMPLE_DIR / "data" / "uniprot.tsv")
    ec_model = make_ec_model(model, adapter, uniprot_db=db)
    assert ec_model.ec.n_enzymes == 5


# --------------------------------------------------------------------------- #
# gecko-light layout
# --------------------------------------------------------------------------- #

def _build_light_ectestgem() -> EcModel:
    model, adapter = _load_fresh_ectestgem()
    return make_ec_model(model, adapter, gecko_light=True)


def test_make_ec_model_gecko_light_sets_flag():
    ec = _build_light_ectestgem()
    assert ec.ec.gecko_light is True


def test_make_ec_model_gecko_light_ec_rxns_have_3digit_prefix():
    """Every ec.rxns id in a light model carries a `###_` counter prefix
    distinguishing isozymes of the same cobra reaction."""
    ec = _build_light_ectestgem()
    for rxn_id in ec.ec.rxns:
        assert (
            len(rxn_id) > 4
            and rxn_id[3] == "_"
            and rxn_id[:3].isdigit()
        ), f"{rxn_id!r} lacks the `###_` light prefix"


def test_make_ec_model_gecko_light_one_row_per_isozyme():
    """ecTestGEM's R2 has two isozymes (G1 and G2 as a complex, plus G3
    alone). Both directions of R2 should appear twice in ec.rxns."""
    ec = _build_light_ectestgem()
    assert "001_R2" in ec.ec.rxns
    assert "002_R2" in ec.ec.rxns
    assert "001_R2_REV" in ec.ec.rxns
    assert "002_R2_REV" in ec.ec.rxns
    # R3 and R5 are single-isozyme; only the 001_ entry appears.
    assert "001_R3" in ec.ec.rxns
    assert "001_R5" in ec.ec.rxns
    assert "002_R3" not in ec.ec.rxns
    assert "002_R5" not in ec.ec.rxns


def test_make_ec_model_gecko_light_coupling_row_per_isozyme():
    """Each ec.rxns row's coupling vector contains 1.0 for exactly the
    genes in that isozyme's AND-clause, not the whole GPR."""
    ec = _build_light_ectestgem()
    gene_idx = {g: i for i, g in enumerate(ec.ec.genes)}
    mat = ec.ec.rxn_enz_mat.toarray()

    # 001_R2 = G1 AND G2 (the complex isozyme)
    i = ec.ec.rxns.index("001_R2")
    expected = np.zeros(ec.ec.n_enzymes)
    expected[gene_idx["G1"]] = 1.0
    expected[gene_idx["G2"]] = 1.0
    np.testing.assert_array_equal(mat[i], expected)

    # 002_R2 = G3 alone
    i = ec.ec.rxns.index("002_R2")
    expected = np.zeros(ec.ec.n_enzymes)
    expected[gene_idx["G3"]] = 1.0
    np.testing.assert_array_equal(mat[i], expected)


def test_make_ec_model_gecko_light_skips_per_enzyme_prot_mets():
    """Light models have only the shared ``prot_pool``; no per-enzyme
    ``prot_<accession>`` pseudometabolites or ``usage_prot_<accession>``
    reactions."""
    ec = _build_light_ectestgem()
    met_ids = {m.id for m in ec.metabolites}
    rxn_ids = {r.id for r in ec.reactions}

    assert "prot_pool" in met_ids
    assert "prot_pool_exchange" in rxn_ids
    assert not any(
        mid.startswith("prot_") and mid != "prot_pool" for mid in met_ids
    )
    assert not any(rid.startswith("usage_prot_") for rid in rxn_ids)


def test_make_ec_model_gecko_light_kcat_initialized_zero():
    ec = _build_light_ectestgem()
    assert np.all(ec.ec.kcat == 0.0)


def test_make_ec_model_gecko_light_enzymes_match_full():
    """The per-enzyme arrays (genes/enzymes/mw/sequence) are identical to
    the full build because stage 7 doesn't branch on gecko_light."""
    model, adapter = _load_fresh_ectestgem()
    full = make_ec_model(model, adapter)
    model2, adapter2 = _load_fresh_ectestgem()
    light = make_ec_model(model2, adapter2, gecko_light=True)

    assert full.ec.genes == light.ec.genes
    assert full.ec.enzymes == light.ec.enzymes
    np.testing.assert_array_equal(full.ec.mw, light.ec.mw)
    assert full.ec.sequence == light.ec.sequence


def test_make_ec_model_missing_uniprot_tsv_raises(tmp_path):
    """If the adapter points at a folder without uniprot.tsv, and no
    explicit uniprot_db is given, raise FileNotFoundError."""
    toml = (
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'enzyme_comp = "c"\n'
    )
    (tmp_path / "model_adapter.toml").write_text(toml)
    adapter = ModelAdapter.from_folder(tmp_path)

    model = cobra.Model("m")
    m = cobra.Metabolite("A", compartment="c")
    model.add_metabolites([m])
    r = cobra.Reaction("r1")
    r.add_metabolites({m: -1})
    r.gene_reaction_rule = "g1"
    model.add_reactions([r])

    with pytest.raises(FileNotFoundError, match="uniprot.tsv"):
        make_ec_model(model, adapter)


def test_make_ec_model_refuses_rerun_on_populated_ecmodel():
    """Calling make_ec_model twice on the same model should raise the
    second time, because the model already has a populated ec."""
    model, adapter = _load_fresh_ectestgem()
    ec_model = make_ec_model(model, adapter)

    with pytest.raises(ValueError, match="already has a populated ec"):
        make_ec_model(ec_model, adapter)


def test_make_ec_model_unmatched_genes_logged_and_annotated(tmp_path, caplog):
    """A model whose gene is absent from the UniProt DB should produce
    a logged warning and per-reaction note, but no return value."""
    toml = (
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'enzyme_comp = "c"\n'
    )
    (tmp_path / "model_adapter.toml").write_text(toml)
    adapter = ModelAdapter.from_folder(tmp_path)

    model = cobra.Model("m")
    m_a = cobra.Metabolite("A", compartment="c")
    m_b = cobra.Metabolite("B", compartment="c")
    model.add_metabolites([m_a, m_b])
    r = cobra.Reaction("r1")
    r.add_metabolites({m_a: -1, m_b: 1})
    r.lower_bound, r.upper_bound = 0.0, 1000.0
    r.gene_reaction_rule = "g_known and g_missing"
    model.add_reactions([r])

    db = UniprotDB(
        ids=["P1"],
        genes=["g_known"],
        eccodes=[""],
        mw=np.array([10.0]),
        sequences=["S"],
    )

    with caplog.at_level(logging.WARNING, logger="geckopy.ec_model.make_ec_model"):
        ec_model = make_ec_model(model, adapter, uniprot_db=db)

    assert "g_missing" in caplog.text
    warned = ec_model.reactions.get_by_id("r1")
    assert "geckopy_warning" in warned.notes
    assert "g_missing" in warned.notes["geckopy_warning"]
