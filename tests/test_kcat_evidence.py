"""Tests for kcat_evidence and kcats_for_reactions.

GECKO's kcatEvidenceTests test the same behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from geckopy.gather_kcats import ec_rxn_keys, kcat_evidence, kcats_for_reactions, kcats_from_okp, save_kcat_source
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
    assert list(result.summary.columns)[:8] == ["ec_rxn", "kcat", "tuned", "fold", "prior", "origin", "enzymes", "eccodes"]
    assert list(result.evidence.columns) == ["ec_rxn", "kcat", "source", "substrate", "detail", "assigned"]
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
    with pytest.raises(FileNotFoundError, match="OKP_job_kinform_h__query.txt"):
        kcats_from_okp(model, method="KinForm-H", job_tag="query", use_stored=True, adapter=adapter)
    with pytest.raises(ValueError, match="no letters or digits"):
        kcats_from_okp(model, method="KinForm-H", job_tag="!!", use_stored=True, adapter=adapter)
    with pytest.raises(ValueError, match="Not in model.ec.rxns"):
        kcats_from_okp(model, method="KinForm-H", ec_rxns=["nope"], adapter=adapter)
    with pytest.raises(ValueError, match="outside model.ec.rxns"):
        kcats_from_okp(model, method="KinForm-H", ec_rxns=[10**6], adapter=adapter)


def test_tagged_okp_submit_leaves_the_untagged_files(tmp_path, monkeypatch):
    from geckopy.gather_kcats.open_kinetics_predictor.run import _submit_open_kinetics_predictor
    from test_open_kinetics_predictor import _FakeResponse, _FakeSession, _adapter, _ec_model, _ignore

    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    session = _FakeSession([("/submit/", _FakeResponse(201, {"jobId": "JOB7", "status": "Pending"}))])
    assert _submit_open_kinetics_predictor(model, ignore_lists=_ignore(), session=session,
                                           job_tag="query", overwrite=True) == "JOB7"
    data = tmp_path / "data"
    assert (data / "OKP_query_query.csv").read_text().startswith("Protein Sequence,Substrate")
    assert not (data / "OKP.csv").exists() and not (data / "OKP_job.txt").exists()


def test_empty_and_string_arguments(tmp_path):
    model, adapter, _ = _project(tmp_path)
    for kwargs in ({"rxns": []}, {"enzymes": []}, {"tuned": []}):
        assert len(kcat_evidence(model, sources=[], adapter=adapter, **kwargs).rows) == 0
    rxn = model.ec.rxns[_rows_with_enzymes(model, 1)[0]]
    result = kcat_evidence(model, tuned=rxn, sources=[], compute="ec_tree",
                           measurements=_measurements_for(model), adapter=adapter)
    assert list(result.summary["ec_rxn"]) == [rxn] and result.summary["src_ec_tree"].notna().all()
    with pytest.raises(ValueError, match="outside model.ec.rxns"):
        kcat_evidence(model, rxns=[10**6], adapter=adapter)
    with pytest.raises(ValueError, match="one per row"):
        kcat_evidence(model, rxns=np.array([True, False]), adapter=adapter)
    with pytest.raises(ValueError, match="No reaction"):
        kcats_for_reactions(model, [], adapter=adapter)


def test_assigned_survives_the_rounding_of_the_prior(tmp_path):
    model, adapter, _ = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    rxn = model.ec.rxns[i]
    model.ec.kcat[i] = 123.456789
    model.ec.source[i] = "custom | tuned=sensitivity (prior=12.3457)"
    _save(model, adapter, "src_a", [rxn], [12.3456789])
    result = kcat_evidence(model, rxns=[rxn], sources=["src_a"], adapter=adapter)
    assert result.evidence["assigned"].all()
    # A tuning status without a prior is tuned, and compared with the current kcat.
    model.ec.source[i] = "custom | tuned=cmaes"
    row = kcat_evidence(model, rxns=[rxn], sources=[], adapter=adapter).summary.iloc[0]
    assert row["tuned"] and np.isnan(row["prior"]) and np.isnan(row["fold"])


def test_a_reverse_direction_is_selected_by_its_conventional_name(tmp_path):
    model, adapter, _ = _project(tmp_path)
    base, _ = ec_rxn_keys(model)
    reverse = [b for b in dict.fromkeys(base) if b.endswith("_REV")]
    if not reverse:
        pytest.skip("the test model has no reverse direction")
    result = kcat_evidence(model, rxns=reverse[:1], sources=[], adapter=adapter)
    assert len(result.rows) and all(base[i] == reverse[0] for i in result.rows)
    forward = kcat_evidence(model, rxns=[reverse[0][:-4]], sources=[], adapter=adapter)
    assert not set(forward.rows) & set(result.rows)


def test_a_tagged_query_result_is_not_a_stored_prediction_of_a_method(tmp_path):
    from geckopy.upgrade import _stored_predictions

    data = tmp_path / "data"
    data.mkdir()
    (data / "OKP_output_unikp.csv").write_text("x\n")
    (data / "OKP_job_unikp.txt").write_text("jobId: a\nmethod: UniKP\n")
    (data / "OKP_output_kinform_h__query.csv").write_text("x\n")
    (data / "OKP_job_kinform_h__query.txt").write_text("jobId: b\nmethod: KinForm-H\n")
    assert _stored_predictions(tmp_path) == ["UniKP"]


def test_scalar_and_numpy_arguments(tmp_path):
    model, adapter, _ = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    assert list(kcat_evidence(model, rxns=i, sources=[], adapter=adapter).rows) == [i]
    assert list(kcat_evidence(model, rxns=np.int64(i), sources=[], compute=None, adapter=adapter).rows) == [i]
    assert list(kcat_evidence(model, rxns=model.ec.rxns[i], sources=[], adapter=adapter).rows) == [i]
    with pytest.raises(ValueError, match="Not in model.ec.rxns"):
        kcats_from_okp(model, method="KinForm-H", ec_rxns=model.ec.rxns[i] + "x", adapter=adapter)
    # A single name is one reaction, not a list of characters.
    with pytest.raises(ValueError, match="selects the reactions of a new job"):
        kcats_from_okp(model, method="KinForm-H", ec_rxns=model.ec.rxns[i], job_tag="q", use_stored=True,
                       adapter=adapter)


def test_the_ec_tree_k_of_the_adapter_and_a_source_list_without_duplicates(tmp_path):
    model, adapter, _ = _project(tmp_path, "[kcat_sources.ec_tree]\nk = 10\n")
    i = _rows_with_enzymes(model, 1)[0]
    m = _measurements_for(model)
    result = kcat_evidence(model, rxns=[i], sources=["ec_tree", "ec_tree"], compute=["ec_tree"],
                           measurements=m, adapter=adapter)
    from geckopy.gather_kcats.ec_tree import ECTree
    expected = 10 ** ECTree(m, k=10).estimate(model.ec.eccodes[i])[0]
    assert result.summary["src_ec_tree"].iloc[0] == pytest.approx(expected)
    assert len(result.evidence) == 1
    empty = kcat_evidence(model, rxns=[], sources=[], compute=["ec_tree"], adapter=adapter)
    assert "src_ec_tree" in empty.summary and len(empty.summary) == 0


def test_a_file_that_is_no_source_is_skipped_in_the_default_list_only(tmp_path):
    model, adapter, project = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    (project / "data" / "kcats" / "notes.tsv").write_text("this is not a source file\n", encoding="utf-8")
    result = kcat_evidence(model, rxns=[i], adapter=adapter)
    assert "src_notes" not in result.summary
    with pytest.raises(Exception):
        kcat_evidence(model, rxns=[i], sources=["notes"], adapter=adapter)


def test_a_method_without_result_leaves_no_old_query_file(tmp_path, monkeypatch):
    model, adapter, project = _project(tmp_path)
    i = _rows_with_enzymes(model, 1)[0]
    stale = project / "data" / "kcats" / "okp_unikp__query.tsv"
    _save(model, adapter, "okp_unikp__query", [model.ec.rxns[i]], [3.0])
    assert stale.is_file()
    monkeypatch.delenv("OKP_API_KEY", raising=False)
    result = kcats_for_reactions(model, [i], okp=["UniKP", "UniKP"], measurements=_measurements_for(model),
                                 adapter=adapter)
    assert not stale.exists() and "src_okp_unikp__query" not in result.summary
    result = kcats_for_reactions(model, [i], okp=np.True_, measurements=_measurements_for(model), adapter=adapter)
    assert len(result.summary) == 1
