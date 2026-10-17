"""Smoke test: loading the ecTestGEM example from disk."""
import importlib.util
from pathlib import Path

import cobra
import pytest

from geckopy import ModelAdapter

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


@pytest.fixture
def adapter() -> ModelAdapter:
    return ModelAdapter.from_folder(EXAMPLE_DIR)


def test_adapter_loads_with_expected_params(adapter):
    assert adapter.params.org_name == "testus testus"
    assert adapter.params.c_source == "E1"
    assert adapter.params.bio_rxn == "R4"
    assert adapter.params.enzyme_comp == "c"
    assert adapter.params.kegg.id == "tst"
    assert adapter.params.uniprot.type == "taxonomy"
    assert adapter.params.uniprot.reviewed is True
    assert adapter.params.complex.taxonomic_id == 123456


def test_adapter_resolves_conv_gem_path(adapter):
    assert adapter.params.conv_gem.is_absolute()
    assert adapter.params.conv_gem.name == "testModel.xml"
    assert adapter.params.conv_gem.is_file()


def test_adapter_finds_brenda_folder_and_phyldist(adapter):
    # Without data/brenda in the project, BRENDA comes from geckopy's
    # toolbox-assets release, cached by raven-toolbox.
    folder = adapter.get_brenda_db_folder()
    assert folder.is_dir()
    assert (folder / "kcat.tsv").is_file()
    assert adapter.get_phyl_dist_path().is_file()


def test_sbml_model_loads_via_cobrapy(adapter):
    model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    assert len(model.reactions) > 0
    assert len(model.metabolites) > 0
    assert len(model.genes) > 0


def test_subclass_identifies_spontaneous_reactions():
    adapter_py = EXAMPLE_DIR / "adapter.py"
    spec = importlib.util.spec_from_file_location("ectestgem_adapter", adapter_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    adapter = module.TestGEMAdapter.from_folder(EXAMPLE_DIR)
    model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    assert adapter.get_spontaneous_reactions(model) == ["R4"]
