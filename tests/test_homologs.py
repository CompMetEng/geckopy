"""Tests for kcats_from_homologs, with DIAMOND on a fixture of known identities.

P4 (reaction R3, substrate m1) has an exact copy h1, measured on m1 at 10 and
40 /s, a copy h2 with 60 of 300 residues changed (80% identity), measured at
100 /s, and an unrelated h3 at 1000 /s. GECKO's homologsTests use the same
fixture (tests/data/homologFixture).
"""
import shutil
from pathlib import Path

import cobra
import pandas as pd
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.gather_kcats import kcats_from_homologs, read_kcat_source
from geckopy.gather_kcats.homologs import compact_name, metabolite_inchikeys
from geckopy.gather_kcats.kcat_assignment import kcat_assignment_config

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
FIXTURE = Path(__file__).parent / "data" / "homologFixture"


def _project(tmp_path: Path, toml_extra: str = ""):
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    for name in ("measurements.tsv", "sequences.fasta", "substrates.tsv"):
        shutil.copy(FIXTURE / name, brenda / name)
    if toml_extra:
        toml = project / "model_adapter.toml"
        toml.write_text(toml.read_text(encoding="utf-8") + "\n" + toml_extra, encoding="utf-8")
    adapter = ModelAdapter.from_folder(project)
    model = make_ec_model(cobra.io.read_sbml_model(str(adapter.params.conv_gem)), adapter)
    seqs = dict(line.split("\n", 1) for line in FIXTURE.joinpath("enzymes.fasta").read_text().split(">")[1:])
    model.ec.sequence = [seqs.get(e, s).strip() for e, s in zip(model.ec.enzymes, model.ec.sequence)]
    return model, adapter, project


def test_compact_name():
    assert compact_name("α-D-Glucose") == "alphadglucose"
    assert compact_name("<i>N</i>-acetyl-L-glutamate") == "nacetyllglutamate"
    assert compact_name("Pyruvic acid") == "pyruvicacid"
    assert compact_name("NAD+") == "nad+"
    assert compact_name("2,3-bisphospho-D-glycerate") == "23bisphosphodglycerate"
    assert compact_name("β-alanine &amp; Café") == "betaalaninecafe"


def test_metabolite_inchikeys(tmp_path):
    model, _, _ = _project(tmp_path)
    substrates = pd.DataFrame({"inchikey": ["IK1", "IK2", "IK3"], "kind": ["chebi", "metanetx", "name"],
                               "value": ["12345", "MNXM9", "m2"]})
    model.metabolites.get_by_id("m1c").annotation["chebi"] = "CHEBI:12345"
    model.metabolites.get_by_id("e1e").annotation["metanetx.chemical"] = ["MNXM9"]
    iks = metabolite_inchikeys(model, substrates)
    assert iks["m1c"] == "IK1" and iks["e1e"] == "IK2" and iks["m2c"] == "IK3"


def test_kcats_from_homologs(tmp_path):
    model, adapter, project = _project(tmp_path)
    kcat_list = kcats_from_homologs(model, adapter=adapter)
    assert list(kcat_list["rxn_id"]) == ["R3"]
    # h1 is the same protein (100% identity): its median alone, 20 /s.
    assert kcat_list["kcat"].iloc[0] == pytest.approx(20.0)
    assert kcat_list["detail"].iloc[0] == "identity=1.000; n=1"
    assert kcat_list["substrates"].iloc[0] == ["m1"]
    records, meta = read_kcat_source(project / "data" / "kcats" / "homologs.tsv")
    assert meta["parameters"] == '{"coverage":0.8,"k":10,"min_identity":0.4}'
    assert meta["data"] == "BRENDA release test"
    assert list(records["rxn"]) == ["R3"] and list(records["substrate"]) == ["m1"]
    # Without h1, the homologs below 95% identity are averaged by identity:
    # h2 (80%) alone.
    seqs = (project / "data" / "brenda" / "sequences.fasta").read_text().split(">")
    (project / "data" / "brenda" / "sequences.fasta").write_text(">" + ">".join(x for x in seqs[1:] if not x.startswith("h1")))
    rest = kcats_from_homologs(model, save=False, adapter=adapter)
    assert rest["kcat"].iloc[0] == pytest.approx(100.0)
    assert rest["detail"].iloc[0] == "identity=0.800; n=1"
    # Above 90% identity nothing is left.
    assert kcats_from_homologs(model, min_identity=0.9, save=False, adapter=adapter).empty


def test_homolog_presets(tmp_path):
    model, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "ec_tree_homologs"\n')
    config = kcat_assignment_config(adapter)
    assert [(st.action, st.sources) for st in config.stages] == [
        ("select", ["custom", "database_top"]), ("combine", ["ec_tree", "homologs"]), ("combine", ["okp"])]
    adapter.params.kcat_assignment.preset = "ec_tree_okp"
    assert kcat_assignment_config(adapter).stages[1].sources == ["ec_tree", "homologs", "okp"]
