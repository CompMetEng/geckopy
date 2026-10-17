"""Tests for convert_custom_kcats, upgrade_ec_model and upgrade_project.

GECKO's upgradeTests test the same.
"""
import shutil
from datetime import date
from pathlib import Path

import cobra
import numpy as np

from geckopy import ModelAdapter, make_ec_model, save_ec_model
from geckopy.cli import main
from geckopy.ec_model.constants import POOL_EXCHANGE_ID
from geckopy.gather_kcats import collect_kcats, kcats_from_brenda, read_kcat_source
from geckopy.get_enzyme_data.ec_from_gem import fill_eccodes_from_gem
from geckopy.upgrade import convert_custom_kcats, upgrade_ec_model, upgrade_project
from geckopy.utilities.load_ec_model import load_ec_model

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
DLKCAT = ("rxns\tgenes\tsubstrates\tsmiles\tsequence\tkcat\n"
          "R2_EXP_1\tG1\tm1\tC\tMRAL\t12.5\nR3\tG4\tm1\tC\tMDFM\t3.25\n")


def _project(tmp_path: Path, preset: str = "gecko_median"):
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    (project / "data" / "DLKcat.tsv").write_text(DLKCAT, encoding="utf-8")
    toml = project / "model_adapter.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + f'\n[kcat_assignment]\npreset = "{preset}"\n', encoding="utf-8")
    adapter = ModelAdapter.from_folder(project)
    model = make_ec_model(cobra.io.read_sbml_model(str(adapter.params.conv_gem)), adapter)
    fill_eccodes_from_gem(model)
    return model, adapter, project


def _gecko3(model):
    old = model.copy()
    for rxn in old.reactions:
        if rxn.id.startswith("usage_prot_") or rxn.id == POOL_EXCHANGE_ID:
            rxn.add_metabolites({m: -2 * c for m, c in rxn.metabolites.items()})
            lb, ub = rxn.bounds
            rxn.bounds = (-ub, -lb)
    return old


def _stoich(model):
    return {r.id: ({m.id: c for m, c in r.metabolites.items()}, r.bounds) for r in model.reactions}


def test_convert_custom_kcats(tmp_path):
    """ecTestGEM's customKcats.tsv: P3 200 and P1 + P2 100 (proteins only),
    then R2_REV and R5 50 (reactions only), which replaces P1 + P2 on R2_REV,
    as apply_custom_kcats does."""
    model, adapter, project = _project(tmp_path)
    path, report = convert_custom_kcats(model, adapter=adapter)
    assert path == project / "data" / "kcats" / "custom.tsv"
    text = path.read_text(encoding="utf-8")
    assert "# kind: curated\n" in text and "# source: custom\n" in text
    records, meta = read_kcat_source(path)
    assert meta["kind"] == "curated"
    rows = set(zip(records["rxn"], records["enzymes"], records["kcat"]))
    assert ("R2_REV", "P1 + P2", 50.0) in rows and ("R2", "P1 + P2", 100.0) in rows
    assert report["rows"] == len(rows)


def test_converted_custom_kcats_give_same_kcats(tmp_path):
    model, adapter, project = _project(tmp_path)
    kcats_from_brenda(model, adapter=adapter)
    a, _, _ = collect_kcats(model, adapter=adapter, write_report=False)
    convert_custom_kcats(model, adapter=adapter)
    b, _, _ = collect_kcats(model, adapter=adapter, write_report=False)
    np.testing.assert_array_equal(b["kcat"].to_numpy(), a["kcat"].to_numpy())
    assert list(b["source"] == "custom") == list(a["source"] == "custom")
    assert (b["source"] == "custom").any()


def test_upgrade_ec_model(tmp_path):
    model, _, _ = _project(tmp_path)
    old = _gecko3(model)
    assert _stoich(old) != _stoich(model)
    changed = upgrade_ec_model(old)
    assert _stoich(old) == _stoich(model)
    assert len(changed) == sum(r.id.startswith("usage_prot_") for r in model.reactions) + 1
    assert upgrade_ec_model(old) == []


def test_upgrade_project(tmp_path, capsys):
    model, adapter, project = _project(tmp_path)
    save_ec_model(_gecko3(model), project / "models" / "ecModel.yml")
    (project / "run.py").write_text("from geckopy import select_kcat_value\nx = select_kcat_value(m, k)\n",
                                    encoding="utf-8")
    dry = upgrade_project(project)
    assert list(dry["step"][:4]) == ["custom kcats", "predictions", "saved ecModel", "project format"]
    assert list(dry["status"][:4]) == ["skipped", "skipped", "to do", "to do"]
    assert list(dry.loc[dry["step"] == "script", "note"]) == [
        "run.py:1: select_kcat_value -> apply_kcat_list", "run.py:2: select_kcat_value -> apply_kcat_list"]

    done = upgrade_project(project, dry_run=False, ec_model=model)
    assert set(done.loc[done["step"].isin(["custom kcats", "predictions", "saved ecModel", "project format"]),
                        "status"]) == {"done"}
    assert (project / "model_adapter.toml").read_text(encoding="utf-8").startswith("project_format = 4\n")
    assert ModelAdapter.from_folder(project).params.project_format == 4
    assert (project / "data" / "kcats" / "custom.tsv").is_file()
    assert (project / "data" / "kcats" / "dlkcat.tsv").is_file()
    legacy = project / "legacy" / date.today().isoformat()
    assert (legacy / "data" / "customKcats.tsv").is_file()
    assert (legacy / "models" / "ecModel.yml").is_file()
    assert upgrade_ec_model(load_ec_model(project / "models" / "ecModel.yml")) == []

    again = upgrade_project(project)
    assert set(again["status"]) == {"by hand"}
    assert main(["upgrade", str(project)]) == 0
    assert "select_kcat_value -> apply_kcat_list" in capsys.readouterr().out
