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
    with pytest.raises(ValueError, match="unknown key sgma in model_adapter.toml"):
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
sigma0_log = 0.4

[evotune.source_groups.brenda]
sources = ["brenda"]
sigma0_log = 0.2

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


def test_uniprot_table_lookup_reads_quoted_cells_and_first_duplicate(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML)
    data_dir = folder / "data"
    data_dir.mkdir()
    (data_dir / "uniprotConversion.tsv").write_text(
        "gene_id\tuniprot_id\n"
        '"YAL001C"\t"P12345"\n'
        "YAL001C\tP00000\n"
        'YAL002W\t"P67890"\r\n'
    )

    adapter = ModelAdapter.from_folder(folder)
    result = adapter.get_uniprot_ids_from_table(["YAL001C", "YAL002W"])
    assert result == ["P12345", "P67890"]


def test_okp_section_has_no_method():
    from geckopy.adapter.params import OkpParams

    assert "method" not in OkpParams.model_fields


@pytest.mark.parametrize("extra, key", [
    ("typo = 1\n", "typo"),
    ("[kegg]\nidd = \"sce\"\n", "kegg.idd"),
    ("[okp]\nmethod = \"CataPro\"\n", "okp.method"),
    ("[evotune]\nmatch_okp = true\n", "evotune.match_okp"),
    ("[evotune.source_groups.brenda]\nsources = [\"brenda\"]\nsigma0_log = 0.2\nmatch_okp = true\n",
     "evotune.source_groups.brenda.match_okp"),
    ("[evotune.sigma0_log_source]\nbrenda = 0.2\n", "evotune.sigma0_log_source"),
    ("[[kcat_assignment.stage]]\nsources = [\"custom\"]\ncriterion = \"max\"\n", "kcat_assignment.stage.criterion"),
    ("[kcat_sources.brenda]\naggregate = \"max\"\n", "kcat_sources.brenda.aggregate"),
    ("[kcat_sources.okp_kinform_h]\nmethod = \"KinForm-H\"\n", "kcat_sources.okp_kinform_h.method"),
    ("[kcat_sources.nosuchsource]\nk = 5\n", "kcat_sources.nosuchsource"),
    ("[matlab]\nmake_model_anaerobicc = \"f\"\n", "matlab.make_model_anaerobicc"),
    ("kcat_aggregate_brenda = \"max\"\n", "kcat_aggregate_brenda"),
])
def test_an_unknown_key_is_an_error_naming_file_and_key(tmp_path, extra, key):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML + extra)
    with pytest.raises(ValueError) as err:
        ModelAdapter.from_folder(folder)
    assert str(err.value) == f"{folder.resolve() / 'model_adapter.toml'}: unknown key {key} in model_adapter.toml"


def test_known_kcat_source_keys_and_free_matlab_params_load(tmp_path):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML + """\
[kcat_sources.brenda]
aggregation = "max"
force_wildcard_level = 1
[kcat_sources.ec_tree]
k = 5
[kcat_sources.homologs]
k = 5
min_identity = 0.4
coverage = 0.5
[kcat_sources.okp]
methods = ["KinForm-H"]
[kcat_sources.okp_kinform_h]
[kcat_sources.custom]
[kcat_sources.dlkcat]
[matlab.params]
anything = 1
""")
    assert ModelAdapter.from_folder(folder).params.kcat_sources["homologs"]["coverage"] == 0.5


SHOW_TOML = MINIMAL_TOML + """\
gr_exp = 0.5

[kcat_sources.brenda]
aggregation = "max"

[evotune.source_groups.okp]
sources = ["okp_*"]
sigma0_log = 0.25
"""


def test_show_prints_one_line_per_item_in_the_agreed_format(tmp_path, capsys):
    from geckopy.adapter import show_adapter_params

    show_adapter_params(ModelAdapter.from_folder(_write_adapter_folder(tmp_path, SHOW_TOML)))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "project_format =  [not set]"
    for expected in [
        "org_name = Testus organismus  [file]",
        "gr_exp = 0.5  [file]",
        "sigma = 0.5  [default]",
        "uniprot.reviewed = false  [default]",
        "kcat_assignment.preset = brenda_ec_tree  [default]",
        "kcat_assignment.isozymes = true  [default]",
        "kcat_assignment.stage[1] = select: custom, database_top, ec_tree; top_origin_limit=1  [preset brenda_ec_tree]",
        "kcat_assignment.stage[2] = combine: okp  [preset brenda_ec_tree]",
        "kcat_sources.brenda.aggregation = max  [file]",
        "kcat_sources.ec_tree.k = 5  [preset brenda_ec_tree]",
        "kcat_sources.okp.methods = KinForm-H  [default]",
        "evotune.source_groups.okp = okp_*; sigma0_log 0.25  [file]",
    ]:
        assert expected in lines, expected
    assert not any(ln.startswith("Adapter parameters") for ln in lines)
    # Items in the order of the table; each table key is listed (stage, kcat_sources, groups expanded).
    table = Path(__file__).resolve().parent.parent / "src" / "geckopy" / "adapter" / "adapterParameters.tsv"
    expected_keys = [ln.split("\t")[0] for ln in table.read_text().splitlines()[1:]]
    seen = []
    for ln in lines:
        key = ln.split(" =")[0]
        base = next((k for k in expected_keys if key == k or key.startswith(k + ".") or key.startswith(k + "[")), key)
        if not seen or seen[-1] != base:
            seen.append(base)
    assert seen == expected_keys


def test_show_with_stages_and_file_settings(tmp_path, capsys):
    from geckopy.adapter import show_adapter_params

    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML + """[[kcat_assignment.stage]]
sources = ["custom", "database_top"]
top_origin_limit = 2
criteria = "median"
[[kcat_assignment.stage]]
sources = ["okp"]
action = "combine"
""")
    show_adapter_params(ModelAdapter.from_folder(folder))
    lines = capsys.readouterr().out.splitlines()
    assert "kcat_assignment.preset = none (stages are given)  [file]" in lines
    assert "kcat_assignment.stage[1] = select: custom, database_top; criteria=median; top_origin_limit=2  [file]" in lines
    assert "kcat_assignment.stage[2] = combine: okp  [file]" in lines


def test_show_cli_prints_the_project_of_the_folder(tmp_path, capsys):
    from geckopy.cli import main

    folder = _write_adapter_folder(tmp_path, SHOW_TOML)
    assert main(["adapter", "show", str(folder)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "project_format =  [not set]"
    assert "gr_exp = 0.5  [file]" in out


def _group_toml(body: str) -> str:
    return MINIMAL_TOML + "[evotune.source_groups.g]\n" + body


@pytest.mark.parametrize("toml, message", [
    ("[evotune.source_groups]\ng = 1\n", "evotune.source_groups.g must be a table with sources and sigma0_log"),
    (_group_toml('sources = ["a"]\n'), "evotune.source_groups.g needs both sources and sigma0_log"),
    (_group_toml("sigma0_log = 0.1\n"), "evotune.source_groups.g needs both sources and sigma0_log"),
    (_group_toml('sources = "a"\nsigma0_log = 0.1\n'), "evotune.source_groups.g.sources must be a non-empty list of strings"),
    (_group_toml('sources = []\nsigma0_log = 0.1\n'), "evotune.source_groups.g.sources must be a non-empty list of strings"),
    (_group_toml('sources = ["a"]\nsigma0_log = inf\n'), "evotune.source_groups.g.sigma0_log must be a positive number"),
    (_group_toml('sources = ["a"]\nsigma0_log = nan\n'), "evotune.source_groups.g.sigma0_log must be a positive number"),
    (_group_toml('sources = ["a", 1]\nsigma0_log = 0.1\n'), "evotune.source_groups.g.sources must be a non-empty list of strings"),
    (_group_toml('sources = ["a"]\nsigma0_log = "x"\n'), "evotune.source_groups.g.sigma0_log must be a positive number"),
    (_group_toml('sources = ["a"]\nsigma0_log = 0\n'), "evotune.source_groups.g.sigma0_log must be a positive number"),
    (_group_toml('sources = ["a"]\nsigma0_log = true\n'), "evotune.source_groups.g.sigma0_log must be a positive number"),
])
def test_a_malformed_evotune_group_has_its_own_message(tmp_path, toml, message):
    folder = _write_adapter_folder(tmp_path, toml)
    with pytest.raises(ValueError) as err:
        ModelAdapter.from_folder(folder)
    assert str(err.value) == f"{folder.resolve() / 'model_adapter.toml'}: {message}"


def test_okp_source_tables_are_limited_to_known_methods(tmp_path):
    ok = _write_adapter_folder(tmp_path, MINIMAL_TOML + "[kcat_sources.okp_unikp]\n")
    assert "okp_unikp" in ModelAdapter.from_folder(ok).params.kcat_sources
    for name in ("okp_nosuchmethod", "okp_catpred"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model_adapter.toml").write_text(MINIMAL_TOML + f"[kcat_sources.{name}]\n")
        with pytest.raises(ValueError, match=f"unknown key kcat_sources.{name} in model_adapter.toml"):
            ModelAdapter.from_folder(folder)


@pytest.mark.parametrize("extra, message", [
    ("[kcat_sources.okp]\nmethods = \"KinForm-H\"\n", "kcat_sources.okp.methods must be a list of strings"),
    ("[kcat_sources.okp]\nmethods = [1]\n", "kcat_sources.okp.methods must be a list of strings"),
    ("kcat_sources.brenda = 1\n", "kcat_sources.brenda must be a table"),
    ("kcat_sources.foo = 1\n", "unknown key kcat_sources.foo in model_adapter.toml"),
    ("[kcat_sources.brenda]\naggregation = \"mean\"\n", "kcat_sources.brenda.aggregation must be max or median"),
])
def test_kcat_source_values_are_checked_at_load(tmp_path, extra, message):
    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML + extra)
    with pytest.raises(ValueError) as err:
        ModelAdapter.from_folder(folder)
    assert str(err.value) == f"{folder.resolve() / 'model_adapter.toml'}: {message}"


def test_unknown_okp_methods_are_an_error_at_load_in_file_order_without_duplicates(tmp_path):
    from geckopy.gather_kcats.kcat_sources import OKP_METHODS

    folder = _write_adapter_folder(
        tmp_path, MINIMAL_TOML + '[kcat_sources.okp]\nmethods = ["Zed", "KinForm-H", "Alpha", "Zed", "TurNup"]\n')
    with pytest.raises(ValueError) as err:
        ModelAdapter.from_folder(folder)
    assert str(err.value) == (
        "Unknown OKP methods in [kcat_sources.okp] methods: Zed, Alpha, TurNup; "
        f"methods: {', '.join(OKP_METHODS)}.")


def test_show_number_formats_and_kcat_sources_order(tmp_path, capsys):
    from geckopy.adapter import show_adapter_params

    folder = _write_adapter_folder(tmp_path, MINIMAL_TOML + """\
sigma = 1e6
[complex]
taxonomic_id = 5592920
[kcat_sources.okp]
methods = ["UniKP"]
[kcat_sources.ec_tree]
k = 3
[kcat_sources.brenda]
force_wildcard_level = 1
aggregation = "median"
[kcat_assignment]
preset = "gecko_classic"
""")
    show_adapter_params(ModelAdapter.from_folder(folder))
    lines = capsys.readouterr().out.splitlines()
    assert "sigma = 1000000  [file]" in lines
    assert "complex.taxonomic_id = 5592920  [file]" in lines
    assert ("kcat_assignment.stage[1] = select: custom, database_top, dlkcat, database_bottom; "
            "top_origin_limit=6  [preset gecko_classic]") in lines
    ks = [ln for ln in lines if ln.startswith("kcat_sources.")]
    assert ks == [
        "kcat_sources.brenda.aggregation = median  [file]",
        "kcat_sources.brenda.force_wildcard_level = 1  [file]",
        "kcat_sources.ec_tree.k = 3  [file]",
        "kcat_sources.okp.methods = UniKP  [file]",
    ]
