"""Tests for load_ec_model."""
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from geckopy import EcModel, ModelAdapter
from geckopy.utilities import load_ec_model


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
    )
    (tmp_path / "models").mkdir(exist_ok=True)
    return ModelAdapter.from_folder(tmp_path)


def _canonical_yaml() -> dict:
    """A minimal but complete canonical-format ecModel doc.

    Topology:
        R1 catalysed by E1 (gene g1)
        R2 catalysed by E2 (gene g2)
        usage_prot_E1, usage_prot_E2, prot_pool_exchange.
    """
    return {
        "id": "demo",
        "name": "demo ecModel",
        "compartments": {"c": "cytosol"},
        "metabolites": [
            {"id": "A_c", "compartment": "c"},
            {"id": "B_c", "compartment": "c"},
            {"id": "C_c", "compartment": "c"},
            {"id": "prot_pool", "compartment": "c"},
            {"id": "prot_E1", "compartment": "c"},
            {"id": "prot_E2", "compartment": "c"},
        ],
        "reactions": [
            {
                "id": "R1",
                "metabolites": {"A_c": -1.0, "prot_E1": -0.01, "B_c": 1.0},
                "lower_bound": 0, "upper_bound": 1000,
                "gene_reaction_rule": "g1",
            },
            {
                "id": "R2",
                "metabolites": {"B_c": -1.0, "prot_E2": -0.01, "C_c": 1.0},
                "lower_bound": 0, "upper_bound": 1000,
                "gene_reaction_rule": "g2",
            },
            {
                "id": "usage_prot_E1",
                "metabolites": {"prot_pool": -1.0, "prot_E1": 1.0},
                "lower_bound": 0, "upper_bound": 1000,
            },
            {
                "id": "usage_prot_E2",
                "metabolites": {"prot_pool": -1.0, "prot_E2": 1.0},
                "lower_bound": 0, "upper_bound": 1000,
            },
            {
                "id": "prot_pool_exchange",
                "metabolites": {"prot_pool": 1.0},
                "lower_bound": 0, "upper_bound": 1000,
            },
        ],
        "genes": [
            {"id": "g1", "name": ""},
            {"id": "g2", "name": ""},
        ],
        "ec-rxns": [
            {
                "id": "R1", "kcat": 1.5, "source": "brenda",
                "eccodes": "1.1.1.1",
                "enzymes": {"E1": 1},
            },
            {
                "id": "R2", "kcat": 2.5, "source": "dlkcat",
                "eccodes": ["2.2.2.2", "2.2.99.99"],
                "enzymes": {"E2": 1},
            },
        ],
        "ec-enzymes": [
            {"genes": "g1", "enzymes": "E1", "mw": 100.0,
             "sequence": "MAA"},
            {"genes": "g2", "enzymes": "E2", "mw": 200.0,
             "sequence": "MBB"},
        ],
        "metaData": {"version": "1", "date": "2026-05-15"},
    }


def _write_yaml(path: Path, doc: dict) -> None:
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)


def _write_canonical(path: Path) -> None:
    _write_yaml(path, _canonical_yaml())


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_relative_filename_without_adapter_raises(tmp_path):
    with pytest.raises(ValueError, match="adapter"):
        load_ec_model("ecModel.yml", adapter=None)


def test_default_filename_without_adapter_raises(tmp_path):
    with pytest.raises(ValueError, match="adapter"):
        load_ec_model(adapter=None)


def test_unknown_extension_raises(tmp_path):
    """Anything outside YAML is rejected (SBML ecModel I/O was removed)."""
    with pytest.raises(ValueError, match="YAML only"):
        load_ec_model("ecModel.txt", adapter=None)


def test_sbml_extension_raises_with_helpful_message(tmp_path):
    """A `.xml`/`.sbml` path raises a specific message naming SBML
    ecModel I/O as removed, rather than the generic "YAML only" error."""
    with pytest.raises(ValueError, match="SBML ecModel I/O has been removed"):
        load_ec_model("ecModel.xml", adapter=None)


def test_missing_file_raises(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ec_model("missing.yml", adapter=adapter)


def test_scalar_top_level_raises(tmp_path):
    adapter = _adapter(tmp_path)
    path = tmp_path / "models" / "ecModel.yml"
    # A genuinely non-mapping document (not a sequence of single-key maps).
    path.write_text("just a scalar\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_ec_model("ecModel.yml", adapter=adapter)


def test_legacy_sequence_top_level_is_merged(tmp_path):
    """A legacy RAVEN `- key: val` sequence-of-single-key-maps loads as
    if it had been written as one big mapping — verifies the merge in
    raven-toolbox's reader is reached through the geckopy wrapper.
    """
    adapter = _adapter(tmp_path)
    # Take the canonical doc and re-emit it as one-key-per-list-item.
    doc = _canonical_yaml()
    path = tmp_path / "models" / "ecModel.yml"
    legacy_lines = []
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    import io as _io
    for key, value in doc.items():
        block = _io.StringIO()
        yaml.dump({key: value}, block)
        legacy_lines.append("- " + block.getvalue().replace("\n", "\n  "))
    path.write_text("\n".join(legacy_lines))
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.id == "demo"
    assert model.ec.rxns == ["R1", "R2"]


def test_missing_ec_rxns_raises(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    del doc["ec-rxns"]
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    with pytest.raises(ValueError, match="ec-rxns"):
        load_ec_model("ecModel.yml", adapter=adapter)


def test_missing_ec_enzymes_raises(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    del doc["ec-enzymes"]
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    with pytest.raises(ValueError, match="ec-enzymes"):
        load_ec_model("ecModel.yml", adapter=adapter)


def test_unknown_enzyme_in_ec_rxn_raises(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["ec-rxns"][0]["enzymes"] = {"E_ghost": 1}
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    with pytest.raises(ValueError, match="E_ghost"):
        load_ec_model("ecModel.yml", adapter=adapter)


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

def test_default_filename_resolved_under_adapter_models(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.id == "demo"


def test_relative_filename_resolved_under_adapter_models(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "myModel.yml")
    model = load_ec_model("myModel.yml", adapter=adapter)
    assert model.id == "demo"


def test_absolute_path_does_not_require_adapter(tmp_path):
    path = tmp_path / "models" / "freestanding.yml"
    path.parent.mkdir()
    _write_canonical(path)
    model = load_ec_model(path, adapter=None)
    assert model.id == "demo"


# --------------------------------------------------------------------------- #
# Returned EcModel structure
# --------------------------------------------------------------------------- #

def test_returns_ec_model_with_adapter_attached(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert isinstance(model, EcModel)
    assert model.adapter is adapter


def test_cobra_section_loaded(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert {r.id for r in model.reactions} >= {
        "R1", "R2", "usage_prot_E1", "usage_prot_E2", "prot_pool_exchange",
    }
    assert {m.id for m in model.metabolites} >= {
        "A_c", "B_c", "C_c", "prot_pool", "prot_E1", "prot_E2",
    }
    assert {g.id for g in model.genes} == {"g1", "g2"}


def test_ec_rxns_fields(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.ec.rxns == ["R1", "R2"]
    np.testing.assert_array_equal(model.ec.kcat, [1.5, 2.5])
    assert model.ec.source == ["brenda", "dlkcat"]
    assert model.ec.notes == ["", ""]


def test_eccodes_scalar_and_list_canonicalized(tmp_path):
    """Scalar and list eccodes both end up as `;`-joined strings."""
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.ec.eccodes == ["1.1.1.1", "2.2.2.2;2.2.99.99"]


def test_ec_enzymes_fields(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.ec.genes == ["g1", "g2"]
    assert model.ec.enzymes == ["E1", "E2"]
    np.testing.assert_array_equal(model.ec.mw, [100.0, 200.0])
    assert model.ec.sequence == ["MAA", "MBB"]
    assert model.ec.concs.shape == (2,)
    assert np.all(np.isnan(model.ec.concs))


def test_rxn_enz_mat_built_from_enzymes_map(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.ec.rxn_enz_mat.shape == (2, 2)
    np.testing.assert_array_equal(
        model.ec.rxn_enz_mat.toarray(), np.eye(2),
    )


# --------------------------------------------------------------------------- #
# Optional fields default correctly
# --------------------------------------------------------------------------- #

def test_missing_kcat_defaults_to_zero(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    del doc["ec-rxns"][0]["kcat"]
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.ec.kcat[0] == 0


def test_missing_mw_defaults_to_nan(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    del doc["ec-enzymes"][0]["mw"]
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert np.isnan(model.ec.mw[0])


def test_missing_source_defaults_to_empty(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    del doc["ec-rxns"][0]["source"]
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.ec.source[0] == ""


def test_missing_concs_defaults_to_nan(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert np.all(np.isnan(model.ec.concs))


def test_explicit_concs_loaded(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["ec-enzymes"][0]["concs"] = 0.005
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.ec.concs[0] == pytest.approx(0.005)
    assert np.isnan(model.ec.concs[1])


# --------------------------------------------------------------------------- #
# gecko_light flag
# --------------------------------------------------------------------------- #

def test_gecko_light_flag_default_false(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model(adapter=adapter)
    assert model.ec.gecko_light is False


def test_gecko_light_flag_true(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["gecko_light"] = True
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.ec.gecko_light is True


# --------------------------------------------------------------------------- #
# Annotations and metaData
# --------------------------------------------------------------------------- #

def test_metadata_silently_ignored(tmp_path):
    """metaData provenance fields don't cause errors and don't leak
    into model.ec or model.annotation in surprising ways."""
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    # Just verify loading succeeds; metaData was set in the fixture.
    load_ec_model(adapter=adapter)


def test_metabolite_annotation_loaded(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["metabolites"][0]["annotation"] = {
        "kegg.compound": ["C00001"],
        "smiles": ["O"],
    }
    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)
    a_c = model.metabolites.get_by_id("A_c")
    assert a_c.annotation.get("kegg.compound") == ["C00001"]
    assert a_c.annotation.get("smiles") == ["O"]


def test_autoflip_legacy_reverse_direction(tmp_path):
    """Older MATLAB ecModels stored usage_prot_* and prot_pool_exchange
    in the reverse direction (lb < 0, ub == 0, stoichiometry signs
    flipped). load_ec_model should detect that on the fly, warn, and
    flip the affected reactions back to the forward convention.
    """
    import warnings

    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    # Mutate the protein reactions into the legacy reverse shape, keeping
    # the effective cap by mapping (lb=0, ub=U) -> (lb=-U, ub=0).
    for rxn in doc["reactions"]:
        if rxn["id"].startswith("usage_prot_") or rxn["id"] == "prot_pool_exchange":
            rxn["lower_bound"] = -rxn["upper_bound"]
            rxn["upper_bound"] = 0.0
            rxn["metabolites"] = {k: -v for k, v in rxn["metabolites"].items()}

    path = tmp_path / "models" / "ecModel.yml"
    _write_yaml(path, doc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flipped = load_ec_model("ecModel.yml", adapter=adapter)

    msgs = [str(w.message) for w in caught]
    assert any("in the GECKO 3 direction" in m and "upgrade_project" in m for m in msgs), (
        f"expected a GECKO 3 warning, got: {msgs}"
    )
    assert not any("forward convention" in m for m in msgs)

    # Every flipped reaction should now look like the canonical fixture.
    canonical_doc = _canonical_yaml()
    expected = {
        r["id"]: (r["lower_bound"], r["upper_bound"], dict(r["metabolites"]))
        for r in canonical_doc["reactions"]
        if r["id"].startswith("usage_prot_") or r["id"] == "prot_pool_exchange"
    }
    for rxn in flipped.reactions:
        if rxn.id not in expected:
            continue
        lb_exp, ub_exp, mets_exp = expected[rxn.id]
        assert rxn.lower_bound == lb_exp, rxn.id
        assert rxn.upper_bound == ub_exp, rxn.id
        got = {m.id: c for m, c in rxn.metabolites.items()}
        assert got == mets_exp, (rxn.id, got, mets_exp)


def test_autoflip_no_warning_when_already_forward(tmp_path):
    """A model that already uses the forward convention must load
    without the flip warning."""
    import warnings

    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_ec_model(adapter=adapter)
    msgs = [str(w.message) for w in caught]
    assert not any("forward convention" in m for m in msgs), (
        f"unexpected flip warning on already-forward model: {msgs}"
    )


# --------------------------------------------------------------------------- #
# Legacy MATLAB / RAVEN layout normalization
# --------------------------------------------------------------------------- #

def test_legacy_metadata_and_smiles_normalized(tmp_path):
    """A legacy-shaped doc (id/name/version nested under metaData,
    per-metabolite smiles as a top-level key) loads correctly: id/name
    are lifted, and smiles ends up under annotation."""
    doc = _canonical_yaml()
    # Move id/name/version into metaData (legacy nesting).
    doc["metaData"] = {
        "id": doc.pop("id"),
        "name": doc.pop("name"),
        "version": "1",
    }
    # Put smiles as a top-level metabolite key (legacy placement).
    doc["metabolites"][0]["smiles"] = "C(C)O"

    adapter = _adapter(tmp_path)
    path = tmp_path / "models" / "ecModel.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(path, doc)
    model = load_ec_model("ecModel.yml", adapter=adapter)

    assert model.id == "demo"
    assert model.name == "demo ecModel"
    met = model.metabolites.get_by_id("A_c")
    assert met.annotation.get("smiles") in ("C(C)O", ["C(C)O"])


def test_newer_ecmodel_format_raises(tmp_path):
    """A file of an ecModel format above 4 is refused, naming its writer."""
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["metaData"] = {"id": "demo", "gecko_format": "5", "generator": "geckopy 9.0.0"}
    _write_yaml(tmp_path / "models" / "ecModel.yml", doc)
    with pytest.raises(ValueError, match=r"ecModel format 5 \(written by geckopy 9\.0\.0\)"):
        load_ec_model("ecModel.yml", adapter=adapter)


def test_format_4_file_loads_without_warning(tmp_path):
    import warnings

    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["metaData"] = {"id": "demo", "gecko_format": "4"}
    _write_yaml(tmp_path / "models" / "ecModel.yml", doc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_ec_model("ecModel.yml", adapter=adapter)
    assert not [w for w in caught if "ecModel" in str(w.message)]


def test_format_that_is_not_a_number_loads(tmp_path):
    """As GECKO's str2double, a gecko_format that is not a number counts as none."""
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["metaData"] = {"id": "demo", "gecko_format": "4.1.0"}
    _write_yaml(tmp_path / "models" / "ecModel.yml", doc)
    load_ec_model("ecModel.yml", adapter=adapter)


def test_newer_format_without_generator_names_unknown_release(tmp_path):
    adapter = _adapter(tmp_path)
    doc = _canonical_yaml()
    doc["metaData"] = {"id": "demo", "gecko_format": "5"}
    _write_yaml(tmp_path / "models" / "ecModel.yml", doc)
    with pytest.raises(ValueError, match="written by an unknown release"):
        load_ec_model("ecModel.yml", adapter=adapter)


def test_other_reader_warnings_pass_through(tmp_path, monkeypatch):
    """Only raven-toolbox's GECKO 3 warning is replaced; other warnings reach the caller."""
    import importlib
    import warnings

    lem = importlib.import_module("geckopy.utilities.load_ec_model")

    adapter = _adapter(tmp_path)
    _write_yaml(tmp_path / "models" / "ecModel.yml", _canonical_yaml())
    read = lem.read_yaml_model

    def read_with_warning(path):
        warnings.warn("unrelated reader warning", UserWarning)
        return read(path)

    monkeypatch.setattr(lem, "read_yaml_model", read_with_warning)
    with pytest.warns(UserWarning, match="unrelated reader warning"):
        load_ec_model("ecModel.yml", adapter=adapter)


# --------------------------------------------------------------------------- #
# Input that MATLAB's readYAMLmodel reads and a YAML parser rejects
# --------------------------------------------------------------------------- #

# The ec reactions as GECKO writes them: R1_EXP_1 has no enzymes (`enzymes: !!omap`
# with no entries, as for a `standard` row), R3 is the last row before the next section.
# The notes of R1_EXP_1 and R3 hold quotes that RAVEN 2 did not escape.
_GECKO_FORM = """\
!!omap
- metaData: !!omap
  - id: "t"
  - name: "t"
- metabolites:
  - !!omap
    - id: "A_c"
    - name: "A"
    - compartment: "c"
- reactions:
  - !!omap
    - id: "R1"
    - name: "r"
    - metabolites: !!omap
      - A_c: -1
    - lower_bound: 0
    - upper_bound: 1000
    - gene_reaction_rule: "g1"
- genes:
  - !!omap
    - id: "g1"
- compartments: !!omap
  - c: "cyto"
- gecko_light: false
- ec-rxns:
  - !!omap
    - id: "R1_EXP_1"
    - kcat: 1.5
    - source: "standard"
    - notes: ""Putative x. And y.""
    - enzymes: !!omap
  - !!omap
    - id: "R2"
    - kcat: 2
    - notes: "say \\"hi\\" now"
    - enzymes: !!omap
        - P1: 1
  - !!omap
    - id: "R3"
    - kcat: 3
    - notes: "say "hi" now \\\\ ok"
    - enzymes: !!omap
- ec-enzymes:
  - !!omap
    - genes: "g1"
    - enzymes: "P1"
    - mw: 10
    - sequence: "ABC"
"""


def test_enzymes_omap_without_entries_and_unescaped_quotes_load(tmp_path):
    """The values are those MATLAB's readYAMLmodel gives: no enzymes for the rows
    without entries, the outer pair of quotes of a note removed and the inner ones kept."""
    path = tmp_path / "gecko.yml"
    path.write_text(_GECKO_FORM, encoding="utf-8")
    model = load_ec_model(path)

    assert model.ec.rxns == ["R1_EXP_1", "R2", "R3"]
    np.testing.assert_array_equal(model.ec.kcat, [1.5, 2.0, 3.0])
    assert model.ec.source == ["standard", "", ""]
    assert model.ec.notes == ['"Putative x. And y."', 'say "hi" now', 'say "hi" now \\ ok']
    assert model.ec.enzymes == ["P1"]
    np.testing.assert_array_equal(model.ec.rxn_enz_mat.toarray(), [[0.0], [1.0], [0.0]])


def test_valid_file_is_not_rewritten(tmp_path):
    """A quoted value with escaped quotes stays as it is."""
    from geckopy.utilities.load_ec_model import _repair_yaml_text

    text = '    - notes: "say \\"hi\\" now"\n    - enzymes: !!omap\n        - P1: 1\n    - notes: ""\n'
    assert _repair_yaml_text(text) == text


def test_file_that_stays_invalid_raises_its_first_error(tmp_path):
    from raven_toolbox.io import read_yaml_model

    path = tmp_path / "bad.yml"
    path.write_text("!!omap\n- id: [unclosed\n", encoding="utf-8")
    with pytest.raises(Exception) as direct:
        read_yaml_model(path)
    with pytest.raises(type(direct.value)):
        load_ec_model(path)


def test_enzymes_omap_without_entries_loads(tmp_path):
    text = "".join(
        line for line in _GECKO_FORM.splitlines(keepends=True)
        if "Putative" not in line and "ok" not in line
    )
    path = tmp_path / "gecko.yml"
    path.write_text(text, encoding="utf-8")
    model = load_ec_model(path)

    assert model.ec.rxns == ["R1_EXP_1", "R2", "R3"]
    assert model.ec.notes == ["", 'say "hi" now', ""]
    np.testing.assert_array_equal(model.ec.rxn_enz_mat.toarray(), [[0.0], [1.0], [0.0]])


def test_loaded_ec_model_has_raven_solver_tolerance(tmp_path):
    adapter = _adapter(tmp_path)
    _write_canonical(tmp_path / "models" / "ecModel.yml")
    model = load_ec_model("ecModel.yml", adapter=adapter)
    assert model.tolerance == 1e-9
