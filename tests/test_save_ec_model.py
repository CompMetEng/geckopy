"""Tests for save_ec_model."""
import os
import re
from datetime import date
from pathlib import Path

import cobra
import numpy as np
import pytest
from ruamel.yaml import YAML
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import load_ec_model, save_ec_model


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_full_ec_model(adapter: ModelAdapter) -> EcModel:
    """A complete two-rxn ecModel with kcat / source / notes /
    eccodes / mw / sequence / concs all populated.
    """
    model = EcModel("demo", adapter=adapter)
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E1 = cobra.Metabolite("prot_E1", compartment="c")
    prot_E2 = cobra.Metabolite("prot_E2", compartment="c")
    model.add_metabolites([A_c, B_c, C_c, pool, prot_E1, prot_E2])

    R1 = cobra.Reaction("R1")
    R1.add_metabolites({A_c: -1.0, prot_E1: -0.01, B_c: 1.0})
    R1.lower_bound = 0.0; R1.upper_bound = 1000.0
    R1.gene_reaction_rule = "g1"

    R2 = cobra.Reaction("R2")
    R2.add_metabolites({B_c: -1.0, prot_E2: -0.01, C_c: 1.0})
    R2.lower_bound = 0.0; R2.upper_bound = 1000.0
    R2.gene_reaction_rule = "g2"

    usage_E1 = cobra.Reaction("usage_prot_E1")
    usage_E1.add_metabolites({pool: -1.0, prot_E1: 1.0})
    usage_E1.lower_bound = 0.0; usage_E1.upper_bound = 1000.0

    usage_E2 = cobra.Reaction("usage_prot_E2")
    usage_E2.add_metabolites({pool: -1.0, prot_E2: 1.0})
    usage_E2.lower_bound = 0.0; usage_E2.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0; pool_ex.upper_bound = 1000.0

    model.add_reactions([R1, R2, usage_E1, usage_E2, pool_ex])

    mat = sparse.lil_matrix((2, 2), dtype=float)
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    model.ec = EcData(
        rxns=["R1", "R2"],
        kcat=np.array([1.5, 2.5]),
        source=["brenda", "dlkcat"],
        notes=["", "manually curated"],
        eccodes=["1.1.1.1", "2.2.2.2;2.2.99.99"],
        genes=["g1", "g2"],
        enzymes=["E1", "E2"],
        mw=np.array([100.0, 200.0]),
        sequence=["MAA", "MBB"],
        concs=np.array([np.nan, 0.005]),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def _read_yaml(path: Path) -> dict:
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_empty_ec_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = EcModel("empty", adapter=adapter)
    with pytest.raises(ValueError, match="ec is empty"):
        save_ec_model(model)


def test_no_enzymes_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.enzymes = []
    with pytest.raises(ValueError, match="ec is empty"):
        save_ec_model(model)


def test_relative_filename_without_adapter_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        save_ec_model(model, "ecModel.yml", adapter=None)


def test_unknown_extension_raises(tmp_path):
    """Any extension other than YAML is rejected."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    with pytest.raises(ValueError, match="YAML only"):
        save_ec_model(model, "ecModel.txt")


def test_sbml_extension_raises_with_helpful_message(tmp_path):
    """A `.xml`/`.sbml` filename raises with a message explaining that
    SBML ecModel I/O is not supported, not just a generic "YAML only"
    error."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    with pytest.raises(ValueError, match="SBML ecModel I/O has been removed"):
        save_ec_model(model, "ecModel.xml")


# --------------------------------------------------------------------------- #
# Path resolution and file creation
# --------------------------------------------------------------------------- #

def test_default_filename_resolved_under_models_subdir(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    written = save_ec_model(model)
    assert written == tmp_path / "models" / "ecModel.yml"
    assert written.is_file()


def test_models_subdir_created_if_missing(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    assert not (tmp_path / "models").exists()
    save_ec_model(model)
    assert (tmp_path / "models").is_dir()


def test_explicit_relative_filename(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    written = save_ec_model(model, "myModel.yaml")
    assert written == tmp_path / "models" / "myModel.yaml"
    assert written.is_file()


def test_absolute_path_does_not_require_adapter(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.adapter = None
    target = tmp_path / "elsewhere" / "x.yml"
    written = save_ec_model(model, target, adapter=None)
    assert written == target
    assert written.is_file()


def test_adapter_argument_overrides_model_adapter(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "other"\n'
    )
    other_adapter = ModelAdapter.from_folder(other_root)
    written = save_ec_model(model, adapter=other_adapter)
    assert written == other_root / "models" / "ecModel.yml"


# --------------------------------------------------------------------------- #
# Doc structure
# --------------------------------------------------------------------------- #

def test_doc_top_level_keys(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    # raven-toolbox's RAVEN/cobra-aligned writer nests the model id under
    # ``metaData`` (alongside date/version/description), so ``id`` is not a
    # top-level key.
    assert {
        "metabolites", "reactions", "genes", "compartments",
        "ec-rxns", "ec-enzymes", "gecko_light", "metaData",
    }.issubset(doc.keys())


def test_metadata_includes_date_version_description(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    md = doc["metaData"]
    assert md["date"] == date.today().isoformat()
    assert md["gecko_format"] == "4"
    assert md["generator"].startswith("geckopy ")
    assert "geckopy_version" not in md
    assert md["description"] == "Enzyme-constrained model of demo"
    # The same lines as GECKO's saveEcModel writes.
    text = path.read_text()
    assert '  - gecko_format: "4"\n' in text
    assert "  - generator: geckopy " in text


def test_metadata_keeps_model_metadata(tmp_path):
    """The model's own metaData, its date included, is kept, as RAVEN keeps model.date and annotation."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.notes["metaData"] = {"date": "2025-01-31", "geckopy_version": "4.0.0b1",
                               "taxonomy": "taxonomy/4932", "organization": "Chalmers"}
    doc = _read_yaml(save_ec_model(model))
    assert doc["metaData"]["date"] == "2025-01-31"
    assert doc["metaData"]["taxonomy"] == "taxonomy/4932"
    assert doc["metaData"]["organization"] == "Chalmers"
    assert "geckopy_version" not in doc["metaData"]


def test_input_model_not_mutated(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    before_attrs = vars(model).copy()
    save_ec_model(model)
    # In particular, we don't add a `description` attribute the way
    # MATLAB's saveEcModel does.
    assert "description" not in vars(model) or vars(model).get(
        "description"
    ) == before_attrs.get("description")


def test_gecko_light_default_false(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert doc["gecko_light"] is False


def test_gecko_light_true_preserved(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.gecko_light = True
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert doc["gecko_light"] is True


# --------------------------------------------------------------------------- #
# Sparse emission of defaults
# --------------------------------------------------------------------------- #

def test_empty_source_omitted_in_ec_rxn(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.source[0] = ""
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert "source" not in doc["ec-rxns"][0]


def test_empty_notes_omitted(tmp_path):
    """ec.notes for R1 starts as "" -> not emitted."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert "notes" not in doc["ec-rxns"][0]
    assert doc["ec-rxns"][1]["notes"] == "manually curated"


def test_nan_concs_omitted_per_enzyme(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert "concs" not in doc["ec-enzymes"][0]
    assert doc["ec-enzymes"][1]["concs"] == pytest.approx(0.005)


def test_nan_mw_omitted(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.mw[0] = np.nan
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert "mw" not in doc["ec-enzymes"][0]


def test_empty_sequence_omitted(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.sequence[0] = ""
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert "sequence" not in doc["ec-enzymes"][0]


def test_zero_kcat_emitted_as_zero(tmp_path):
    """Unset kcats (0, the sentinel) are written as ``kcat: 0`` so the
    file round-trips back to 0."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.kcat[0] = 0
    path = save_ec_model(model)
    text = path.read_text()
    # The first ec-rxns entry's kcat line should read "kcat: 0".
    assert "kcat: 0" in text


# --------------------------------------------------------------------------- #
# eccodes scalar / list serialization
# --------------------------------------------------------------------------- #

def test_eccodes_single_serialized_as_scalar(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert doc["ec-rxns"][0]["eccodes"] == "1.1.1.1"


def test_eccodes_multiple_serialized_as_list(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    doc = _read_yaml(path)
    assert doc["ec-rxns"][1]["eccodes"] == ["2.2.2.2", "2.2.99.99"]


# --------------------------------------------------------------------------- #
# Round-trip via load_ec_model
# --------------------------------------------------------------------------- #

def test_round_trip_preserves_ec_rxns(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    assert loaded.ec.rxns == model.ec.rxns
    np.testing.assert_array_equal(loaded.ec.kcat, model.ec.kcat)
    assert loaded.ec.source == model.ec.source
    assert loaded.ec.notes == model.ec.notes
    assert loaded.ec.eccodes == model.ec.eccodes


def test_round_trip_preserves_ec_enzymes(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    assert loaded.ec.genes == model.ec.genes
    assert loaded.ec.enzymes == model.ec.enzymes
    np.testing.assert_array_equal(loaded.ec.mw, model.ec.mw)
    assert loaded.ec.sequence == model.ec.sequence
    np.testing.assert_array_equal(loaded.ec.concs, model.ec.concs)


def test_round_trip_preserves_rxn_enz_mat(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    np.testing.assert_array_equal(
        loaded.ec.rxn_enz_mat.toarray(),
        model.ec.rxn_enz_mat.toarray(),
    )


def test_round_trip_preserves_cobra_structure(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    assert {r.id for r in loaded.reactions} == {
        r.id for r in model.reactions
    }
    assert {m.id for m in loaded.metabolites} == {
        m.id for m in model.metabolites
    }
    assert {g.id for g in loaded.genes} == {g.id for g in model.genes}


def test_round_trip_with_zero_kcat(tmp_path):
    """An unset (=0) kcat round-trips back to 0 through save+load."""
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.kcat[0] = 0
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    assert loaded.ec.kcat[0] == 0
    assert loaded.ec.kcat[1] == pytest.approx(2.5)


def test_round_trip_with_gecko_light_true(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_full_ec_model(adapter)
    model.ec.gecko_light = True
    path = save_ec_model(model)
    loaded = load_ec_model(path)
    assert loaded.ec.gecko_light is True


# --------------------------------------------------------------------------- #
# ec reaction without enzymes
# --------------------------------------------------------------------------- #

def _build_model_with_enzymeless_rxn(adapter: ModelAdapter) -> EcModel:
    """`_build_full_ec_model` with R2 as an ec reaction without enzymes."""
    model = _build_full_ec_model(adapter)
    mat = model.ec.rxn_enz_mat.tolil()
    mat[1, 1] = 0.0
    model.ec.rxn_enz_mat = mat.tocsr()
    model.ec.rxn_enz_mat.eliminate_zeros()
    model.ec.source[1] = "standard"
    return model


def test_rxn_without_enzymes_has_no_enzymes_entry(tmp_path):
    adapter = _adapter(tmp_path)
    path = save_ec_model(_build_model_with_enzymeless_rxn(adapter))
    text = path.read_text(encoding="utf-8")
    ec_rxns = text.split("- ec-rxns:")[1].split("- ec-enzymes:")[0]
    assert ec_rxns.count("enzymes") == 1
    r2 = re.split(r'- id: "?R2"?\n', ec_rxns)[1]
    assert "enzymes" not in r2
    assert "standard" in r2
    assert "!!omap []" not in text
    doc = _read_yaml(path)
    assert "enzymes" not in doc["ec-rxns"][1]


def test_round_trip_rxn_without_enzymes(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_enzymeless_rxn(adapter)
    loaded = load_ec_model(save_ec_model(model))
    assert loaded.ec.rxns == ["R1", "R2"]
    assert loaded.ec.source == ["brenda", "standard"]
    np.testing.assert_array_equal(loaded.ec.kcat, model.ec.kcat)
    np.testing.assert_array_equal(
        loaded.ec.rxn_enz_mat.toarray(), [[1.0, 0.0], [0.0, 0.0]],
    )
    assert loaded.ec.enzymes == ["E1", "E2"]


_LIGHT_ECMODEL = Path(os.environ.get("LIGHT_ECMODEL_YML", "/nonexistent"))


@pytest.mark.skipif(
    not _LIGHT_ECMODEL.is_file(),
    reason="set LIGHT_ECMODEL_YML to GECKO's tutorials/light_ecModel/models/ecModel.yml",
)
def test_light_tutorial_model_loses_only_the_empty_enzymes_entries(tmp_path):
    """The light tutorial ecModel, written by GECKO 3, has `enzymes: !!omap` with
    no entries for its reactions without enzymes. Loaded and saved again, the
    file has no `enzymes` entry for them and the ec data is unchanged."""
    loaded = load_ec_model(_LIGHT_ECMODEL)
    n_zero = int((loaded.ec.rxn_enz_mat.getnnz(axis=1) == 0).sum())
    assert n_zero == 52
    path = save_ec_model(loaded, tmp_path / "out.yml", adapter=None)
    saved = path.read_text(encoding="utf-8")
    saved_rxns = saved.split("- ec-rxns:")[1].split("- ec-enzymes:")[0]
    assert "!!omap []" not in saved_rxns
    assert saved_rxns.count("- id:") == len(loaded.ec.rxns)
    assert saved_rxns.count("enzymes:") == len(loaded.ec.rxns) - n_zero
    again = load_ec_model(path)
    assert again.ec.rxns == loaded.ec.rxns
    np.testing.assert_array_equal(again.ec.kcat, loaded.ec.kcat)
    assert again.ec.source == loaded.ec.source
    assert again.ec.enzymes == loaded.ec.enzymes
    assert (again.ec.rxn_enz_mat != loaded.ec.rxn_enz_mat).nnz == 0
