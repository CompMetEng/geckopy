"""Tests for kcats_from_wildkcat."""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import (
    ec_rxn_keys,
    kcats_from_wildkcat,
    load_kcat_source,
    read_kcat_source,
    save_kcat_source,
    update_kcat_sources,
)

ENZYMES = ["P1", "P2", "P3", "P4"]
# model.ec reactions of the full model and the enzymes of each.
FULL = {"R2_EXP_1": ["P1", "P2"], "R2_REV_EXP_2": ["P2"], "R3": ["P3", "P4"]}
LIGHT = {"001_R2": ["P1", "P2"], "002_R2_REV": ["P2"], "003_R3": ["P3", "P4"]}


def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text('conv_gem = "dummy.xml"\norg_name = "testus testus"\n', encoding="utf-8")
    return ModelAdapter.from_folder(tmp_path)


def _model(adapter: ModelAdapter, light: bool) -> EcModel:
    model = EcModel("test", adapter=adapter, gecko_light=light)
    reactions = LIGHT if light else FULL
    mat = sparse.lil_matrix((len(reactions), len(ENZYMES)))
    for i, enzymes in enumerate(reactions.values()):
        for e in enzymes:
            mat[i, ENZYMES.index(e)] = 1
    n = len(reactions)
    model.ec = EcData(
        gecko_light=light, rxns=list(reactions), kcat=np.zeros(n), source=[""] * n, notes=[""] * n,
        eccodes=["1.1.1.1"] * n, genes=[f"G{k}" for k in range(len(ENZYMES))], enzymes=list(ENZYMES),
        mw=np.full(len(ENZYMES), 40000.0), sequence=["M"] * len(ENZYMES),
        concs=np.full(len(ENZYMES), np.nan), rxn_enz_mat=mat.tocsr(),
    )
    return model


def _genes(model: EcModel, rxn: str) -> str:
    i = model.ec.rxns.index(rxn)
    cols = model.ec.rxn_enz_mat.tocsr()[i].indices
    return ";".join(sorted({model.ec.genes[j] for j in cols}))


def test_wildkcat_import(tmp_path):
    adapter = _adapter(tmp_path)
    full, light = _model(adapter, light=False), _model(adapter, light=True)
    base, _ = ec_rxn_keys(full)
    other = next(i for i, b in enumerate(base) if not b.startswith("R2"))
    header = ["rxn", "ec_code", "direction", "genes", "uniprot", "kcat", "db", "penalty_score",
              "kcat_substrate", "kcat_organism", "kcat_temperature", "kcat_ph", "kcat_id_percent"]
    rows = [
        ["R2", "1.1.1.1", "forward", _genes(full, "R2_EXP_1"), "", "10", "brenda", "3", "A", "Homo sapiens", "30", "7", "80"],
        ["R2", "1.1.1.1", "forward", _genes(full, "R2_EXP_1"), "", "20", "sabio_rk", "0", "A", "Homo sapiens", "25", "7.5", "100"],
        ["R2", "1.1.1.1", "reverse", _genes(full, "R2_REV_EXP_2"), "", "5", "brenda", "6", "B", "", "", "", ""],
        [base[other], "", "forward", _genes(full, full.ec.rxns[other]), "", "7", "catapro", "16", "", "", "", "", ""],
        ["R2", "", "forward", "noSuchGene", "", "9", "brenda", "0", "", "", "", "", ""],
        ["R2", "", "forward", _genes(full, "R2_EXP_1"), "", "", "", "16", "", "", "", "", ""],
    ]
    file = tmp_path / "kcat_full.tsv"
    file.write_text("\n".join("\t".join(r) for r in [header] + rows) + "\n", encoding="utf-8")

    kcat_list, report = kcats_from_wildkcat(full, file, save=False)
    assert report["rows"] == 6
    assert report["no_value"] == 1
    assert report["excluded"] == 0
    assert report["matched"] == 4
    assert report["unmatched"] == ["R2|forward|noSuchGene"]
    kcats = dict(zip(kcat_list["rxn_id"], kcat_list["kcat"]))
    details = dict(zip(kcat_list["rxn_id"], kcat_list["detail"]))
    assert kcats["R2_EXP_1"] == 20
    assert kcats["R2_REV_EXP_2"] == 5
    assert details["R2_EXP_1"] == "db=sabio_rk; penalty=0; organism=Homo sapiens; temperature=25; pH=7.5; identity=100"
    assert full.ec.rxns[other] in kcats

    kcat_list, report = kcats_from_wildkcat(full, file, save=False, max_penalty=4, predictions=False)
    assert report["excluded"] == 2
    assert sorted(kcat_list["rxn_id"]) == ["R2_EXP_1"]

    out = tmp_path / "out"
    kcat_list, _ = kcats_from_wildkcat(full, file, save=False)
    saved = save_kcat_source(full, kcat_list, "wildkcat", folder=out)
    loaded, loaded_report = load_kcat_source(light, saved)
    assert loaded_report["unmatched"].empty
    assert len(loaded["kcat"]) == 3
    with pytest.raises(TypeError):
        kcats_from_wildkcat(full, file, maxpenalty=1)


def test_wildkcat_saves_source_file(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    file = tmp_path / "kcat_retrieved.tsv"
    file.write_text("rxn\tdirection\tgenes\tkcat\tdb\tpenalty_score\n"
                    f"R2\tforward\t{_genes(full, 'R2_EXP_1')}\t4.5\tbrenda\t1\n", encoding="utf-8")
    kcats_from_wildkcat(full, file)
    records, meta = read_kcat_source(tmp_path / "data" / "kcats" / "wildkcat.tsv")
    assert meta["generator"].startswith("kcats_from_wildkcat")
    assert meta["parameters"] == '{"file":"kcat_retrieved.tsv","predictions":true}'
    assert list(records["kcat"]) == [4.5]
    kcats_from_wildkcat(full, file, max_penalty=2, predictions=False, save_as="wk2")
    _, meta = read_kcat_source(tmp_path / "data" / "kcats" / "wk2.tsv")
    assert meta["parameters"] == '{"file":"kcat_retrieved.tsv","max_penalty":2,"predictions":false}'


def test_wildkcat_input_errors(tmp_path):
    full = _model(_adapter(tmp_path), light=False)
    with pytest.raises(FileNotFoundError):
        kcats_from_wildkcat(full, tmp_path / "missing.tsv", save=False)
    bad = tmp_path / "bad.tsv"
    bad.write_text("rxn\tdirection\tkcat\nR2\tforward\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="genes"):
        kcats_from_wildkcat(full, bad, save=False)


def test_update_kcat_sources_skips_wildkcat(tmp_path):
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "testus testus"\n'
        '[[kcat_assignment.stage]]\nsources = ["wildkcat"]\n', encoding="utf-8")
    adapter = ModelAdapter.from_folder(tmp_path)
    report = update_kcat_sources(_model(adapter, light=False), dry_run=True)
    row = report[report["source"] == "wildkcat"].iloc[0]
    assert row["status"] == "skipped"
    assert row["note"] == "imported from a WILDkCAT run with kcats_from_wildkcat"
