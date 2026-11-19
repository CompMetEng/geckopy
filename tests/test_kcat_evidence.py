"""Tests for kcat_evidence and kcats_for_reactions.

GECKO's kcatEvidenceTests test the same behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from geckopy.gather_kcats import kcat_evidence, kcats_for_reactions, kcats_from_okp, save_kcat_source
from test_ec_tree import _measurements_for, _project


def _rows_with_enzymes(model, n):
    mat = model.ec.rxn_enz_mat.tocsr()
    return [i for i in range(len(model.ec.rxns))
            if mat[i].nnz and str(model.ec.eccodes[i] or "").strip()][:n]


def _save(model, adapter, name, rxns, kcats):
    save_kcat_source(model, pd.DataFrame({"rxn_id": rxns, "kcat": kcats}), name, adapter=adapter)


def test_evidence_lists_every_source(tmp_path):
    model, adapter, _ = _project(tmp_path)
    rows = _rows_with_enzymes(model, 3)
    sel = [model.ec.rxns[i] for i in rows]
    _save(model, adapter, "src_a", sel, [1.0, 2.0, 4.0])
    _save(model, adapter, "src_b", sel[:1], [8.0])
    result = kcat_evidence(model, rxns=sel, sources=["src_a", "src_b"], adapter=adapter)
    assert list(result.rows) == rows
    assert list(result.summary["ec_rxn"]) == sel
    assert list(result.summary["src_src_a"]) == [1.0, 2.0, 4.0]
    assert result.summary["src_src_b"].iloc[0] == 8.0 and result.summary["src_src_b"].iloc[1:].isna().all()
    assert len(result.evidence) == 4
    assert list(result.summary["kcat"]) == list(np.asarray(model.ec.kcat)[rows])
    only_b = kcat_evidence(model, rxns=rows, sources=["src_b"], adapter=adapter)
    assert "src_src_a" not in only_b.summary
    with pytest.raises(ValueError, match="Not reactions of model.ec"):
        kcat_evidence(model, rxns=["nope"], adapter=adapter)
    with pytest.raises(ValueError, match="Cannot compute"):
        kcat_evidence(model, compute=["x"], adapter=adapter)


def test_evidence_of_a_tuned_kcat(tmp_path):
    model, adapter, _ = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    rxn = model.ec.rxns[i]
    model.ec.kcat[i] = 5.0
    model.ec.source[i] = "custom | tuned=sensitivity (prior=0.5)"
    _save(model, adapter, "src_a", [rxn], [0.5])
    out = tmp_path / "summary.tsv"
    result = kcat_evidence(model, tuned=[rxn], sources=["src_a"], save_as=out, adapter=adapter)
    row = result.summary.iloc[0]
    assert (row["kcat"], row["prior"], row["fold"], row["origin"], row["tuned"]) == (5.0, 0.5, 10.0, "custom", True)
    assert result.evidence["assigned"].all()
    assert pd.read_csv(out, sep="\t")["fold"].iloc[0] == pytest.approx(10.0)


def test_evidence_by_enzyme_and_ec_tree(tmp_path):
    model, adapter, _ = _project(tmp_path)
    j = int(np.flatnonzero(np.asarray((model.ec.rxn_enz_mat != 0).sum(axis=0)).ravel())[0])
    expected = sorted(model.ec.rxn_enz_mat.tocsc()[:, j].nonzero()[0].tolist())
    result = kcat_evidence(model, enzymes=[model.ec.enzymes[j]], sources=[], compute=["ec_tree"],
                           measurements=_measurements_for(model), adapter=adapter)
    assert sorted(result.rows.tolist()) == expected
    with_ec = [i for i in expected if str(model.ec.eccodes[i] or "").strip()]
    assert int(result.summary["src_ec_tree"].notna().sum()) == len(with_ec)
    assert set(result.evidence["source"]) == {"ec_tree"}
    with pytest.raises(ValueError, match="Not in model.ec.enzymes"):
        kcat_evidence(model, enzymes=["X00000"], adapter=adapter)


def test_kcats_for_reactions_without_okp(tmp_path):
    model, adapter, _ = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    result = kcats_for_reactions(model, [i], measurements=_measurements_for(model), adapter=adapter)
    assert list(result.summary["ec_rxn"]) == [model.ec.rxns[i]]
    assert result.summary["src_ec_tree"].notna().all()
    with pytest.raises(ValueError, match="Unknown OKP methods"):
        kcats_for_reactions(model, [i], okp="NoSuchMethod", adapter=adapter)


def test_okp_job_tag_has_its_own_files(tmp_path):
    model, adapter, project = _project(tmp_path)
    data = project / "data"
    (data / "OKP_job_kinform_h.txt").write_text("jobId: abc\nmethod: KinForm-H\nresultOf: abc\n", encoding="utf-8")
    (data / "OKP_output_kinform_h.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="OKP_job_kinform_h_query.txt"):
        kcats_from_okp(model, method="KinForm-H", job_tag="query", use_stored=True, adapter=adapter)
