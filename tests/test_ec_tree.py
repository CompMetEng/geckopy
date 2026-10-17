"""Tests for the EC tree, kcats_from_ec_tree and the brenda_ec_tree preset.

GECKO's ecTreeTests test the same numbers.
"""
import shutil
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.databases import load_brenda_measurements
from geckopy.gather_kcats import (
    assign_kcats,
    collect_kcats,
    kcats_from_brenda,
    kcats_from_ec_tree,
    read_kcat_source,
    update_kcat_sources,
)
from geckopy.gather_kcats.ec_tree import ECTree
from geckopy.get_enzyme_data.ec_from_gem import fill_eccodes_from_gem

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


def _table() -> pd.DataFrame:
    """log10 kcats: global median 2; class 1.1 median 1 (5 values), 1.1.1 median 1.5 (4)."""
    rows = [("1.1.1.1", 0), ("1.1.1.1", 1), ("1.1.1.1", 2), ("1.1.1.2", 3), ("1.1.2.1", -1),
            ("2.7.1.1", 2), ("2.7.1.1", 2)]
    t = pd.DataFrame(rows, columns=["ec", "log10_kcat"])
    t["kcat"] = 10.0 ** t["log10_kcat"]
    t.attrs["release"] = "BRENDA release test"
    return t


def test_ec_tree_worked_example():
    tree = ECTree(_table(), k=5)
    assert tree.global_median == 2
    # 1.1 (n 5, median 1): 0.5*1 + 0.5*2 = 1.5; 1.1.1 (n 4, median 1.5): 1.5;
    # 1.1.1.1 (n 3, median 1): 3/8*1 + 5/8*1.5 = 1.3125.
    value, n, depth, ec = tree.estimate("1.1.1.1")
    assert value == pytest.approx(1.3125) and (n, depth, ec) == (3, 4, "1.1.1.1")
    assert tree.estimate("1.1.1.9")[:3] == (pytest.approx(1.5), 0, 3)
    assert tree.estimate("1.1.1.-")[:3] == (pytest.approx(1.5), 0, 3)
    # The EC number with most measurements wins; the first on a tie.
    assert tree.estimate("1.1.1.2; 1.1.1.1")[3] == "1.1.1.1"
    assert tree.estimate("1.1.1.9;1.1.1.8")[3] == "1.1.1.9"
    assert tree.estimate("9.9.9.9")[2] == 0
    assert tree.estimate("")[:3] == (2.0, 0, 0)
    # A larger k leans more on the classes: 1.1 gives 1/3*1 + 2/3*2 = 5/3,
    # 1.1.1 gives 2/7*1.5 + 5/7*5/3 = 34/21, 1.1.1.1 gives 3/13*1 + 10/13*34/21.
    assert ECTree(_table(), k=10).estimate("1.1.1.1")[0] == pytest.approx(403 / 273)
    with pytest.raises(ValueError):
        ECTree(_table(), k=0)


def test_bundled_measurement_table():
    folder = Path(__file__).parents[1] / "src" / "geckopy" / "data" / "brenda"
    t = load_brenda_measurements(folder)
    assert t.attrs["release"].startswith("BRENDA release ")
    assert len(t) > 50000 and (t["kcat"] > 0).all()
    assert not t["ec"].str.contains("-").any()


def _project(tmp_path: Path, toml_extra: str = ""):
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    if toml_extra:
        toml = project / "model_adapter.toml"
        toml.write_text(toml.read_text(encoding="utf-8") + "\n" + toml_extra, encoding="utf-8")
    adapter = ModelAdapter.from_folder(project)
    model = make_ec_model(cobra.io.read_sbml_model(str(adapter.params.conv_gem)), adapter)
    fill_eccodes_from_gem(model)
    return model, adapter, project


def _measurements_for(model) -> pd.DataFrame:
    """A measurement table with values for the model's EC numbers."""
    ecs = sorted({e.strip() for c in model.ec.eccodes for e in str(c or "").split(";") if e.strip()})
    rows = [(ec, float(i + 1)) for i, ec in enumerate(ecs) for _ in range(2)]
    t = pd.DataFrame(rows, columns=["ec", "log10_kcat"])
    t["kcat"] = 10.0 ** t["log10_kcat"]
    t.attrs["release"] = "BRENDA release test"
    return t


def test_kcats_from_ec_tree_writes_file(tmp_path):
    model, adapter, project = _project(tmp_path)
    table = _measurements_for(model)
    kcat_list = kcats_from_ec_tree(model, measurements=table, adapter=adapter)
    with_ec = [r for r, c in zip(model.ec.rxns, model.ec.eccodes) if str(c or "").strip()]
    assert set(kcat_list["rxn_id"]) == set(with_ec) and len(with_ec) > 0
    tree = ECTree(table)
    for rxn, kcat in zip(kcat_list["rxn_id"], kcat_list["kcat"]):
        assert np.log10(kcat) == pytest.approx(tree.estimate(model.ec.eccodes[model.ec.rxns.index(rxn)])[0])
    records, meta = read_kcat_source(project / "data" / "kcats" / "ec_tree.tsv")
    assert meta["parameters"] == '{"k":5}'
    assert meta["data"] == "BRENDA release test"
    assert set(records["source"]) == {"ec_tree"}
    assert records["detail"].str.match(r"^n=\d+; depth=[234]$").all()
    # Without a measurement table in the project, geckopy's own is used.
    kcats_from_ec_tree(model, adapter=adapter, save_as="ec_tree_bundled")
    _, meta = read_kcat_source(project / "data" / "kcats" / "ec_tree_bundled.tsv")
    assert meta["data"].startswith("BRENDA release ")


def test_default_preset_is_brenda_ec_tree(tmp_path):
    model, adapter, _ = _project(tmp_path)
    from geckopy.gather_kcats.kcat_assignment import kcat_assignment_config

    config = kcat_assignment_config(adapter)
    assert config.preset == "brenda_ec_tree"
    assert config.stages[0].sources == ["custom", "database_top", "ec_tree", "dlkcat"]
    assert config.stages[0].top_origin_limit == 1
    assert config.sources["ec_tree"] == {"k": 5}


def test_brenda_ec_tree_preset(tmp_path):
    """custom.tsv, then BRENDA at origin 1, then the EC tree; nothing from
    other BRENDA tiers."""
    model, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "brenda_ec_tree"\n')
    table = _measurements_for(model)
    with open(project / "data" / "brenda" / "measurements.tsv", "w", encoding="utf-8") as f:
        f.write("# BRENDA release test\n")
        table[["ec", "kcat"]].to_csv(f, sep="\t", index=False)
    report = update_kcat_sources(model, dry_run=True, adapter=adapter)
    assert list(report["source"]) == ["custom", "brenda", "ec_tree", "dlkcat"]
    brenda = kcats_from_brenda(model, aggregation="median", adapter=adapter)
    kcats_from_ec_tree(model, adapter=adapter)
    decisions, rep, decided = collect_kcats(model, adapter=adapter, write_report=False)
    assert rep["preset"] == "brenda_ec_tree"
    origin1 = set(brenda.loc[(brenda["origin"] == 1) & (brenda["wildcard_level"] == 0), "rxn_id"])
    for rxn, src, stage in zip(decisions["ec_rxn"], decisions["source"], decisions["stage"]):
        if stage != "1":
            continue
        if src.startswith("brenda"):
            assert rxn in origin1 and "origin=1" in src
        else:
            assert src in ("custom", "ec_tree", "dlkcat")
    assert (decisions["source"] == "ec_tree").any()
    # The ec_tree file records k = 5 and the project's BRENDA release: current.
    report = update_kcat_sources(model, dry_run=True, adapter=adapter).set_index("source")["status"]
    assert report["ec_tree"] == "current"
    kcats_from_ec_tree(model, k=2, adapter=adapter)
    report = update_kcat_sources(model, dry_run=True, adapter=adapter).set_index("source")["status"]
    assert report["ec_tree"] == "outdated"
    _, rep = assign_kcats(model, adapter=adapter, write_report=False)
    assert rep["preset"] == "brenda_ec_tree"
