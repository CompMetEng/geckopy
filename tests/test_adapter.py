"""Tests for the ModelAdapter and ModelParameters."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from geckopy import ModelAdapter

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


MINIMAL_TOML = """\
conv_gem = "models/test.xml"
org_name = "Testus organismus"
"""

FULL_TOML = """\
conv_gem = "models/test.xml"
org_name = "Saccharomyces cerevisiae"
sigma = 0.5
p_tot = 0.5
f = 0.4461
gr_exp = 0.41

[kegg]
id = "sce"

[uniprot]
type = "taxonomy"
id = "559292"
reviewed = true

[complex]
taxonomic_id = 559292
"""


def _write_adapter_folder(tmp_path: Path, toml_content: str) -> Path:
    """Create a temporary adapter folder with the given TOML content."""
    (tmp_path / "model_adapter.toml").write_text(toml_content)
    return tmp_path


def test_from_folder_with_minimal_toml(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)
    adapter = ModelAdapter.from_folder(folder)

    assert adapter.params.org_name == "Testus organismus"
    assert adapter.params.sigma == 0.5  # default
    assert adapter.params.path == folder.resolve()
    assert adapter.params.conv_gem == (folder / "models/test.xml").resolve()


def test_from_folder_with_full_toml(tmp_path):
    folder = _write_adapter_folder(tmp_path, FULL_TOML)
    adapter = ModelAdapter.from_folder(folder)

    assert adapter.params.org_name == "Saccharomyces cerevisiae"
    assert adapter.params.f == 0.4461
    assert adapter.params.kegg.id == "sce"
    assert adapter.params.uniprot.reviewed is True
    assert adapter.params.uniprot.id == "559292"


def test_missing_toml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelAdapter.from_folder(tmp_path)


def test_unknown_field_rejected(tmp_path):
    bad_toml = MINIMAL_TOML + '\nsgma = 0.3\n'  # typo
    folder = _write_adapter_folder(tmp_path, bad_toml)
    with pytest.raises(ValidationError):
        ModelAdapter.from_folder(folder)


def test_missing_required_field_rejected(tmp_path):
    # org_name is required, leaving it out should fail
    bad_toml = 'conv_gem = "models/test.xml"\n'
    folder = _write_adapter_folder(tmp_path, bad_toml)
    with pytest.raises(ValidationError):
        ModelAdapter.from_folder(folder)


def test_default_overridable_methods(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)
    adapter = ModelAdapter.from_folder(folder)

    # Defaults should behave like the MATLAB base class
    assert adapter.get_spontaneous_reactions(model=None) == []
    assert adapter.get_uniprot_compatible_genes(["YAL001C"]) == ["YAL001C"]


def test_uniprot_table_lookup_without_file(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)
    adapter = ModelAdapter.from_folder(folder)

    result = adapter.get_uniprot_ids_from_table(["YAL001C", "YAL002W"])
    assert result == ["YAL001C", "YAL002W"]


def test_uniprot_table_lookup_with_file(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)
    data_dir = folder / "data"
    data_dir.mkdir()
    (data_dir / "uniprotConversion.tsv").write_text(
        "gene_id\tuniprot_id\n"
        "YAL001C\tP12345\n"
        "YAL002W\tP67890\n"
    )

    adapter = ModelAdapter.from_folder(folder)
    result = adapter.get_uniprot_ids_from_table(["YAL001C", "YAL002W", "YAL999W"])
    assert result == ["P12345", "P67890", "YAL999W"]  # unknown gene passes through


def test_subclass_from_folder_returns_subclass(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)

    class CustomAdapter(ModelAdapter):
        def get_spontaneous_reactions(self, model):
            return ["spont_rxn_1"]

    adapter = CustomAdapter.from_folder(folder)
    assert isinstance(adapter, CustomAdapter)
    assert adapter.get_spontaneous_reactions(model=None) == ["spont_rxn_1"]


@pytest.mark.parametrize(
    "example_dir", [p for p in EXAMPLES_DIR.iterdir() if p.is_dir()],
    ids=lambda p: p.name,
)
def test_shipped_example_adapter_toml_is_valid(example_dir):
    """Verify every folder under examples/ has a model_adapter.toml that
    validates against the current ModelParameters schema."""
    ModelAdapter.from_folder(example_dir)


SPONTANEOUS_TOML = MINIMAL_TOML + """
[spontaneous]
name_contains = "spontaneous"
file = "data/spontaneousReactions.tsv"
"""


def test_spontaneous_reactions_from_toml(tmp_path):
    import cobra

    folder = _write_adapter_folder(tmp_path, SPONTANEOUS_TOML)
    (folder / "data").mkdir()
    (folder / "data" / "spontaneousReactions.tsv").write_text(
        "rxns\tspontaneous\nR1\t0\nR2\t1\nR3\t1\n"
    )
    model = cobra.Model("m")
    for rid, name in [("R1", "a"), ("R2", "b"), ("R4", "spontaneous c"), ("R5", "d")]:
        model.add_reactions([cobra.Reaction(rid, name=name)])
    adapter = ModelAdapter.from_folder(folder)
    assert adapter.params.spontaneous.file == folder / "data" / "spontaneousReactions.tsv"
    assert adapter.get_spontaneous_reactions(model) == ["R2", "R4"]


# A file as GECKO's convertAdapterToToml writes it, MATLAB-only settings included.
CONVERTED_TOML = """\
# Model adapter for testus testus, converted from TestGEMAdapter.
conv_gem = "models/testModel.xml"
org_name = "testus testus"
sigma = 0.5
p_tot = 0.5
f = 4
gr_exp = 0.41
c_source = "E1"
bio_rxn = "R4"
enzyme_comp = "c"

[kegg]
id = "tst"
gene_id = "kegg"

[uniprot]
type = "taxonomy"
id = "TE00000000"
gene_id_field = "gene_tst"
reviewed = true

[complex]
taxonomic_id = 123456

[evotune.source_groups.dlkcat]
sources = ["dlkcat"]

[evotune.source_groups.brenda]
sources = ["brenda"]

[evotune.sigma0_log_source]
dlkcat = 0.4
brenda = 0.2

[matlab]
make_model_anaerobic = "anaerobicModel_GECKO"

[matlab.params]
c_UptakeExp = 0.641339301
"""


def test_toml_converted_from_matlab_adapter_loads(tmp_path):
    folder = _write_adapter_folder(tmp_path, CONVERTED_TOML)
    adapter = ModelAdapter.from_folder(folder)
    assert adapter.params.f == 4
    assert adapter.params.matlab.make_model_anaerobic == "anaerobicModel_GECKO"
    assert adapter.params.matlab.params == {"c_UptakeExp": 0.641339301}
    assert set(adapter.params.evotune.source_groups) == {"dlkcat", "brenda"}


def _schema_leaves(model_cls, prefix=""):
    """Dotted keys of every leaf field of a pydantic model, with defaults."""
    from pydantic import BaseModel

    leaves = {}
    instance = model_cls.model_construct()
    for name, field in model_cls.model_fields.items():
        key = f"{prefix}{name}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            leaves.update(_schema_leaves(annotation, key + "."))
        else:
            leaves[key] = field
    return leaves


def test_adapter_parameter_table_matches_schema():
    """adapterParameters.tsv, shared with GECKO, lists every key and its default."""
    import csv
    import tomllib

    from geckopy.adapter.params import ModelParameters

    table = Path(__file__).resolve().parent.parent / "src" / "geckopy" / "adapter" / "adapterParameters.tsv"
    with open(table, newline="", encoding="utf-8") as f:
        rows = {r["toml_key"]: r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}

    leaves = _schema_leaves(ModelParameters)
    leaves.pop("path")  # set from the project folder, never written
    leaves.pop("matlab.params")  # free-form
    assert set(rows) == set(leaves)

    for key, field in leaves.items():
        default_text = rows[key]["default"]
        if field.is_required() or not default_text:
            continue
        expected = tomllib.loads(f"x = {default_text}")["x"]
        default = field.get_default(call_default_factory=True)
        assert default == expected, key
