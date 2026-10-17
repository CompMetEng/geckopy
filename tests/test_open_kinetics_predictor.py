"""Tests for the OpenKineticsPredictor (OKP) subpackage."""
from __future__ import annotations

from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.databases import DLKcatIgnoreLists
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats.open_kinetics_predictor import (
    OKPClient,
    OKPError,
    build_okp_input_csv,
    fetch_open_kinetics_predictor,
    parse_okp_output,
    submit_open_kinetics_predictor,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _ec_model(adapter=None) -> EcModel:
    """Two reactions sharing one enzyme + substrate (so a single OKP
    prediction fans out to two ec.rxns), plus a second substrate.

    r1: A(sub) + B(sub) -> C, catalysed by gene g1 (seq S1)
    r2: A(sub) -> D,        catalysed by gene g1 (seq S1)
    """
    model = EcModel("test", adapter=adapter)
    mets = {}
    for mid, name, smi in [
        ("A", "ala", "CCO"), ("B", "beta", "CCN"),
        ("C", "gamma", ""), ("D", "delta", ""),
    ]:
        m = cobra.Metabolite(mid, compartment="c")
        m.name = name
        if smi:
            m.annotation["smiles"] = smi
        mets[mid] = m
    model.add_metabolites(list(mets.values()))

    specs = [
        ("r1", {"A": -1.0, "B": -1.0, "C": 1.0}),
        ("r2", {"A": -1.0, "D": 1.0}),
    ]
    for rid, stoich in specs:
        rxn = cobra.Reaction(rid)
        rxn.lower_bound, rxn.upper_bound = 0.0, 1000.0
        rxn.add_metabolites({mets[k]: v for k, v in stoich.items()})
        model.add_reactions([rxn])

    n, g = 2, 1
    mat = sparse.lil_matrix((n, g), dtype=float)
    mat[0, 0] = 1.0
    mat[1, 0] = 1.0
    model.ec = EcData(
        rxns=["r1", "r2"],
        kcat=np.full(n, np.nan),
        source=[""] * n,
        notes=[""] * n,
        eccodes=[""] * n,
        genes=["g1"],
        enzymes=["g1"],
        mw=np.zeros(g),
        sequence=["S1"],
        concs=np.full(g, np.nan),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _ignore() -> DLKcatIgnoreLists:
    return DLKcatIgnoreLists(ignore_names=[], ignore_smiles=[], currency_pairs=[])


_OKP_RESULT_CSV = (
    "kcat (1/s),Source kcat,Extra Info kcat,Protein Sequence,Substrate,"
    "mean similarity to CataPro training data,max similarity to CataPro training data\n"
    "12.5,Prediction from CataPro,,S1,CCO,61.5,61.5\n"
    "3.2,BRENDA,,S1,CCN,,\n"
)


# Fake requests layer -------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _FakeSession:
    """Returns queued responses keyed by URL substring."""
    def __init__(self, routes):
        self.routes = routes  # list of (substr, response)
        self.calls = []

    def _match(self, url):
        for substr, resp in self.routes:
            if substr in url:
                return resp
        raise AssertionError(f"no fake route for {url}")

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._match(url)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._match(url)


# --------------------------------------------------------------------------- #
# build_okp_input_csv
# --------------------------------------------------------------------------- #

def test_build_input_header_and_pairs():
    csv_text = build_okp_input_csv(_ec_model(), _ignore())
    lines = csv_text.strip().split("\n")
    assert lines[0] == "Protein Sequence,Substrate"
    body = sorted(lines[1:])
    assert body == ["S1,CCN", "S1,CCO"]


def test_build_input_dedups_pairs():
    # A appears as substrate in both r1 and r2 with the same enzyme S1,
    # so (S1, CCO) must appear only once.
    csv_text = build_okp_input_csv(_ec_model(), _ignore())
    assert csv_text.count("S1,CCO") == 1


def test_build_input_only_with_smiles_drops_missing():
    csv_text = build_okp_input_csv(_ec_model(), _ignore(), only_with_smiles=True)
    # C and D have no SMILES and are products anyway; only A,B substrates remain.
    assert "None" not in csv_text


# --------------------------------------------------------------------------- #
# parse_okp_output
# --------------------------------------------------------------------------- #

def test_parse_maps_back_to_ec_rxns():
    df = parse_okp_output(_ec_model(), _OKP_RESULT_CSV)
    # S1+CCO (A) -> consumed by r1 and r2 -> 2 rows; S1+CCN (B) -> r1 -> 1 row.
    assert len(df) == 3
    assert set(df["rxn_id"]) == {"r1", "r2"}


def test_parse_maps_back_from_a_reloaded_models_list_valued_smiles():
    """cobra returns annotation values as lists for a model read from a
    file, which is what `save_ec_model` + `load_ec_model` produces. The
    SMILES index has to key on the string inside the list, not on the
    list itself."""
    model = _ec_model()
    for met in model.metabolites:
        if met.annotation.get("smiles"):
            met.annotation["smiles"] = [met.annotation["smiles"]]

    df = parse_okp_output(model, _OKP_RESULT_CSV)
    assert len(df) == 3
    assert set(df["rxn_id"]) == {"r1", "r2"}
    assert all(s in (["ala"], ["beta"]) for s in df["substrates"])


def test_parse_source_provenance_stripped_and_verbatim():
    df = parse_okp_output(_ec_model(), _OKP_RESULT_CSV)
    sources = set(df["source"])
    # "Prediction from CataPro" -> "CataPro"; "BRENDA" kept verbatim; both
    # tagged with the OKP- pipeline-stage prefix.
    assert "OKP-CataPro" in sources
    assert "OKP-BRENDA" in sources
    assert not any(s.startswith("Prediction from") for s in sources)


def test_parse_kcat_and_columns():
    df = parse_okp_output(_ec_model(), _OKP_RESULT_CSV)
    assert list(df.columns) == [
        "rxn_id", "source", "eccode", "substrates", "genes",
        "kcat", "wildcard_level", "origin",
    ]
    cco_rows = df[df["source"] == "OKP-CataPro"]
    assert all(cco_rows["kcat"] == 12.5)
    assert all(s == ["ala"] for s in cco_rows["substrates"])
    assert all(g == ["g1"] for g in cco_rows["genes"])


def test_parse_no_numeric_kcat_raises():
    csv_text = (
        "kcat (1/s),Source kcat,Extra Info kcat,Protein Sequence,Substrate\n"
        "NA,Prediction from CataPro,,S1,CCO\n"
    )
    with pytest.raises(ValueError, match="no numeric kcat"):
        parse_okp_output(_ec_model(), csv_text)


def test_parse_catrange_class_ranges():
    """CatRange's column holds class ranges; each becomes its geometric
    midpoint, as OKP returned it before."""
    csv_text = (
        "Predicted kcat range: kcat (1/s),Source kcat,Extra Info kcat,Protein Sequence,Substrate\n"
        "10 to 100 s^-1,Prediction from CatRange,Predicted range: 10 to 100 s^-1,S1,CCO\n"
    )
    df = parse_okp_output(_ec_model(), csv_text)
    assert not df.empty
    assert all(df["source"] == "OKP-CatRange")
    assert df["kcat"].iloc[0] == pytest.approx(10 ** 1.5)


@pytest.mark.parametrize("text, expected", [
    ("12.5", 12.5),
    ("10 to 100 s^-1", 10 ** 1.5),
    ("0 to 1e-08 s^-1", 5e-9),
    ("1000 to 1e+08 s^-1", 10 ** 5.5),
    ("", None),
    ("NA", None),
])
def test_kcat_value(text, expected):
    from geckopy.gather_kcats.open_kinetics_predictor.parse_output import _kcat_value

    value = _kcat_value(text)
    if expected is None:
        assert value is None or value != value
    else:
        assert value == pytest.approx(expected)


def test_parse_missing_column_raises():
    csv_text = "kcat (1/s),Protein Sequence\n1.0,S1\n"
    with pytest.raises(ValueError, match="missing expected column"):
        parse_okp_output(_ec_model(), csv_text)


def test_parse_unmapped_rows_raise(caplog):
    csv_text = (
        "kcat (1/s),Source kcat,Extra Info kcat,Protein Sequence,Substrate\n"
        "9.9,Prediction from CataPro,,UNKNOWN_SEQ,CCO\n"
    )
    with pytest.raises(ValueError, match="could be mapped"):
        parse_okp_output(_ec_model(), csv_text)


# --------------------------------------------------------------------------- #
# OKPClient
# --------------------------------------------------------------------------- #

def test_client_submit_posts_form_and_returns_json():
    resp = _FakeResponse(201, {"jobId": "ABC", "status": "Pending"})
    session = _FakeSession([("/submit/", resp)])
    client = OKPClient("ak_key", session=session)
    out = client.submit(b"csv", method="CataPro")
    assert out["jobId"] == "ABC"
    method, url, kwargs = session.calls[0]
    assert method == "POST" and url.endswith("/submit/")
    assert kwargs["headers"]["Authorization"] == "Bearer ak_key"
    assert kwargs["data"]["methods"] == '{"kcat": "CataPro"}'
    assert kwargs["data"]["targets"] == '["kcat"]'


def test_client_status_and_result():
    session = _FakeSession([
        ("/status/", _FakeResponse(200, {"status": "Completed"})),
        ("/result/", _FakeResponse(200, text="kcat (1/s)\n1.0\n")),
    ])
    client = OKPClient("ak_key", session=session)
    assert client.status("J")["status"] == "Completed"
    assert client.result_csv("J").startswith("kcat")


def test_client_raises_on_http_error():
    session = _FakeSession([
        ("/status/", _FakeResponse(404, {"error": "No job found with id 'J'."})),
    ])
    client = OKPClient("ak_key", session=session)
    with pytest.raises(OKPError, match="No job found"):
        client.status("J")


def test_client_methods_no_auth_needed():
    session = _FakeSession([("/methods/", _FakeResponse(200, {"methods": {}}))])
    client = OKPClient("ak_key", session=session)
    assert client.methods() == {"methods": {}}


# --------------------------------------------------------------------------- #
# submit / fetch orchestrators
# --------------------------------------------------------------------------- #

def test_submit_writes_csv_and_metadata(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    session = _FakeSession([
        ("/submit/", _FakeResponse(201, {"jobId": "JOB42", "status": "Pending"})),
    ])
    job_id = submit_open_kinetics_predictor(
        model, ignore_lists=_ignore(), session=session,
    )
    assert job_id == "JOB42"
    data_dir = tmp_path / "data"
    assert (data_dir / "OKP.csv").read_text().startswith("Protein Sequence,Substrate")
    meta = (data_dir / "OKP_job.txt").read_text()
    assert "jobId: JOB42" in meta
    assert "method: CataPro" in meta


def test_submit_missing_key_raises(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.delenv("OKP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No OpenKineticsPredictor API key"):
        submit_open_kinetics_predictor(model, ignore_lists=_ignore())


def test_fetch_completed_downloads_and_parses(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "OKP_job.txt").write_text("jobId: JOB42\nmethod: CataPro\n")
    session = _FakeSession([
        ("/status/", _FakeResponse(200, {"status": "Completed"})),
        ("/result/", _FakeResponse(200, text=_OKP_RESULT_CSV)),
    ])
    done, df = fetch_open_kinetics_predictor(model, session=session)
    assert done is True
    assert len(df) == 3
    assert (tmp_path / "data" / "OKP_output.csv").is_file()


def test_fetch_running_without_wait_returns_false(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "OKP_job.txt").write_text("jobId: JOB42\n")
    session = _FakeSession([
        ("/status/", _FakeResponse(200, {"status": "Processing",
                                         "progress": {"predictionsMade": 0,
                                                      "predictionsTotal": 2}})),
    ])
    done, df = fetch_open_kinetics_predictor(model, session=session)
    assert done is False
    assert df is None


def test_fetch_failed_raises(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "OKP_job.txt").write_text("jobId: JOB42\n")
    session = _FakeSession([("/status/", _FakeResponse(200, {"status": "Failed"}))])
    with pytest.raises(OKPError, match="failed"):
        fetch_open_kinetics_predictor(model, session=session)


def test_fetch_use_stored_parses_saved_file(tmp_path):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "OKP_output.csv").write_text(_OKP_RESULT_CSV)
    done, df = fetch_open_kinetics_predictor(model, True)
    assert done is True
    assert len(df) == 3


def test_fetch_use_stored_missing_file_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    with pytest.raises(FileNotFoundError, match="no stored result"):
        fetch_open_kinetics_predictor(model, True)


def test_fetch_reads_job_id_from_metadata(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    model = _ec_model(adapter)
    monkeypatch.setenv("OKP_API_KEY", "ak_test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "OKP_job.txt").write_text("jobId: FROMFILE\nmethod: CataPro\n")
    session = _FakeSession([
        ("/status/FROMFILE/", _FakeResponse(200, {"status": "Completed"})),
        ("/result/FROMFILE/", _FakeResponse(200, text=_OKP_RESULT_CSV)),
    ])
    done, df = fetch_open_kinetics_predictor(model, session=session)
    assert done is True
    assert any("FROMFILE" in url for _, url, _ in session.calls)
