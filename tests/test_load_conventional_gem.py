"""Tests for load_conventional_gem."""
from pathlib import Path

import cobra
import pytest

from geckopy import ModelAdapter
from geckopy.utilities import load_conventional_gem


def _adapter_with_conv_gem(tmp_path: Path, conv_gem_name: str) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "{conv_gem_name}"\n'
        f'org_name = "test"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def test_loads_yaml_gem(tmp_path):
    """A minimal cobra-style YAML GEM loads."""
    yml_path = tmp_path / "tiny.yml"
    yml_path.write_text(
        "id: tiny\n"
        "name: tiny\n"
        "compartments: {c: cytosol}\n"
        "metabolites:\n"
        "  - {id: A_c, compartment: c}\n"
        "reactions:\n"
        "  - id: R1\n"
        "    metabolites: {A_c: -1.0}\n"
        "    lower_bound: 0\n"
        "    upper_bound: 1000\n"
        "    gene_reaction_rule: ''\n"
        "genes: []\n"
    )
    adapter = _adapter_with_conv_gem(tmp_path, "tiny.yml")
    model = load_conventional_gem(adapter)
    assert isinstance(model, cobra.Model)
    assert model.id == "tiny"
    assert "R1" in {r.id for r in model.reactions}



def test_yaml_gem_keeps_raven_eccodes(tmp_path):
    """RAVEN YAML stores EC numbers as a per-reaction ``eccodes`` field;
    they end up in ``annotation['ec-code']``, where
    ``fill_eccodes_from_gem`` reads them."""
    yml_path = tmp_path / "raven.yml"
    yml_path.write_text(
        "!!omap\n"
        "- metaData: !!omap\n"
        "  - id: raven\n"
        "  - name: raven\n"
        "- metabolites:\n"
        "  - !!omap\n"
        "    - id: A_c\n"
        "    - compartment: c\n"
        "- reactions:\n"
        "  - !!omap\n"
        "    - id: R1\n"
        "    - metabolites: !!omap\n"
        "      - A_c: -1\n"
        "    - lower_bound: 0\n"
        "    - upper_bound: 1000\n"
        "    - eccodes:\n"
        "      - \"1.1.2.4\"\n"
        "      - \"1.1.99.-\"\n"
        "- genes: []\n"
        "- compartments: !!omap\n"
        "  - c: cytosol\n"
    )
    adapter = _adapter_with_conv_gem(tmp_path, "raven.yml")
    model = load_conventional_gem(adapter)
    assert model.reactions.R1.annotation["ec-code"] == ["1.1.2.4", "1.1.99.-"]

def test_loads_sbml_gem(tmp_path):
    """An SBML GEM (the existing ecTestGEM fixture) loads."""
    sbml_src = (
        Path(__file__).parent.parent
        / "examples" / "ecTestGEM" / "models" / "testModel.xml"
    )
    if not sbml_src.is_file():
        pytest.skip("ecTestGEM fixture not present")
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "{sbml_src.as_posix()}"\n'
        'org_name = "test"\n'
    )
    adapter = ModelAdapter.from_folder(tmp_path)
    model = load_conventional_gem(adapter)
    assert isinstance(model, cobra.Model)
    assert len(model.reactions) > 0


def test_missing_file_raises(tmp_path):
    adapter = _adapter_with_conv_gem(tmp_path, "absent.yml")
    with pytest.raises(FileNotFoundError, match="conv_gem"):
        load_conventional_gem(adapter)


def test_loaded_gem_has_raven_solver_tolerance(tmp_path):
    """The model carries RAVEN's solver tolerance whichever reader was used."""
    json_path = tmp_path / "tiny.json"
    cobra.io.save_json_model(cobra.Model("tiny"), str(json_path))
    adapter = _adapter_with_conv_gem(tmp_path, "tiny.json")
    assert load_conventional_gem(adapter).tolerance == 1e-9
