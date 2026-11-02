"""Tests for kcat source files and the kcats_from_* functions."""
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import (
    ec_rxn_keys,
    kcats_from_brenda,
    kcats_from_dlkcat,
    kcats_from_okp,
    load_kcat_source,
    read_kcat_source,
    save_kcat_source,
)
from geckopy.gather_kcats import kcats_from as kcats_from_module
from geckopy.gather_kcats.kcat_sources import format_number

# Conventional reactions with their isozymes (each a set of enzymes).
REACTIONS = {
    "R1": [["P2", "P1"], ["P3"]],
    "R1_REV": [["P2", "P1"], ["P3"]],
    "R2": [["P4"]],
}
ENZYMES = ["P1", "P2", "P3", "P4"]


def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text('conv_gem = "dummy.xml"\norg_name = "testus testus"\n', encoding="utf-8")
    return ModelAdapter.from_folder(tmp_path)


def _model(adapter: ModelAdapter, light: bool) -> EcModel:
    """Full model: R1_EXP_1, R1_EXP_2, ...; light model: 001_R1, 002_R1, ..."""
    model = EcModel("test", adapter=adapter, gecko_light=light)
    met = cobra.Metabolite("m1", compartment="c", name="m1")
    model.add_metabolites([met])
    rxns, rows = [], []
    for base, isozymes in REACTIONS.items():
        for n, enzymes in enumerate(isozymes, start=1):
            if light:
                rxns.append(f"{n:03d}_{base}")
            else:
                rxns.append(f"{base}_EXP_{n}" if len(isozymes) > 1 else base)
            rows.append([ENZYMES.index(e) for e in enzymes])
        if not light:
            for n in range(len(isozymes)):
                rid = f"{base}_EXP_{n + 1}" if len(isozymes) > 1 else base
                model.add_reactions([cobra.Reaction(rid)])
                model.reactions.get_by_id(rid).add_metabolites({met: -1})
        else:
            model.add_reactions([cobra.Reaction(base)])
    mat = sparse.lil_matrix((len(rxns), len(ENZYMES)))
    for i, cols in enumerate(rows):
        for j in cols:
            mat[i, j] = 1
    n = len(rxns)
    model.ec = EcData(
        gecko_light=light, rxns=rxns, kcat=np.zeros(n), source=[""] * n, notes=[""] * n,
        eccodes=["1.1.1.1"] * n, genes=[f"G{k}" for k in range(len(ENZYMES))], enzymes=list(ENZYMES),
        mw=np.full(len(ENZYMES), 40000.0), sequence=["M"] * len(ENZYMES),
        concs=np.full(len(ENZYMES), np.nan), rxn_enz_mat=mat.tocsr(),
    )
    return model


def _kcat_list(model: EcModel) -> pd.DataFrame:
    n = len(model.ec.rxns)
    return pd.DataFrame({
        "rxn_id": model.ec.rxns,
        "source": ["brenda"] * n,
        "eccode": ["1.1.1.1"] * n,
        "substrates": [["m1", "ATP"]] * n,
        "genes": [[] for _ in range(n)],
        "kcat": [10.0 * (k + 1) for k in range(n)],
        "wildcard_level": [0] * n,
        "origin": [k % 6 + 1 for k in range(n)],
    })


def test_ec_rxn_keys_full_and_light(tmp_path):
    adapter = _adapter(tmp_path)
    base_f, enz_f = ec_rxn_keys(_model(adapter, light=False))
    base_l, enz_l = ec_rxn_keys(_model(adapter, light=True))
    assert base_f[:2] == ["R1", "R1"] and base_f[2] == "R1_REV"
    assert enz_f[0] == "P1 + P2"
    assert sorted(zip(base_f, enz_f)) == sorted(zip(base_l, enz_l))


def test_save_read_load_round_trip(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    kcats = _kcat_list(full)
    path = save_kcat_source(full, kcats, "brenda", parameters={"aggregation": "max"}, data="BRENDA release test")
    assert path == tmp_path / "data" / "kcats" / "brenda.tsv"
    records, meta = read_kcat_source(path)
    assert meta["source"] == "brenda" and meta["kind"] == "gathered"
    assert meta["parameters"] == '{"aggregation":"max"}'
    assert records["substrate"].iloc[0] == "m1; ATP"
    assert records["detail"].iloc[0] == "wc=0; origin=1"
    loaded, report = load_kcat_source(full, "brenda")
    assert report["unmatched"].empty
    merged = loaded.merge(kcats, on="rxn_id", suffixes=("", "_in"))
    assert len(merged) == len(kcats)
    assert (merged["kcat"] == merged["kcat_in"]).all()
    assert (merged["origin"] == merged["origin_in"]).all()


def test_full_file_applies_to_light_model(tmp_path):
    adapter = _adapter(tmp_path)
    full, light = _model(adapter, light=False), _model(adapter, light=True)
    save_kcat_source(full, _kcat_list(full), "brenda")
    loaded, report = load_kcat_source(light, "brenda")
    assert report["unmatched"].empty
    assert sorted(loaded["rxn_id"]) == sorted(light.ec.rxns)


def test_matching_rules_and_curated_file(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    folder = tmp_path / "data" / "kcats"
    folder.mkdir(parents=True)
    (folder / "custom.tsv").write_text(
        "# source: custom\n# kind: curated\n"
        "rxn\tenzymes\tsubstrate\tkcat\tlog10_sd\tsource\tdetail\tec_code\tnote\n"
        "\tP2 + P1\t\t10\t\tcustom\t\t\tenzymes only, other order\n"
        "R2\t\t\t20\t\tcustom\t\t\treaction only\n"
        "R999\tX1\t\t30\t\tcustom\t\t\tunmatched\n"
        "R1\tP1 + P2 + XNEW\t\t40\t\tcustom\t\t\tpartial\n"
    , encoding="utf-8")
    loaded, report = load_kcat_source(full, "custom")
    assert sorted(loaded.loc[loaded["line"] == 4, "rxn_id"]) == ["R1_EXP_1", "R1_REV_EXP_1"]
    assert list(loaded.loc[loaded["line"] == 5, "rxn_id"]) == ["R2"]
    assert list(report["unmatched"]["line"]) == [6, 7]
    assert set(report["partial"]["line"]) == {7}
    # Curated files are never written: by name, 'kind: curated' and
    # 'source: custom', in any case; the file is left as it was.
    before = (folder / "custom.tsv").read_bytes()
    with pytest.raises(PermissionError):
        save_kcat_source(full, _kcat_list(full), "custom")
    assert (folder / "custom.tsv").read_bytes() == before
    (folder / "mine.tsv").write_text("# source: Custom\nrxn\tkcat\tsource\n", encoding="utf-8")
    (folder / "mine2.tsv").write_text("# kind: Curated\nrxn\tkcat\tsource\n", encoding="utf-8")
    for name in ("mine", "mine2"):
        with pytest.raises(PermissionError):
            save_kcat_source(full, _kcat_list(full), name)


@pytest.mark.parametrize("value, text", [
    (10.0, "10"), (0.1, "0.1"), (1.5e-05, "1.5e-05"), (123.456, "123.456"),
    (np.nan, ""), (2.0 / 3.0, "0.6666666666666666"), (np.inf, "Inf"), (-np.inf, "-Inf"),
    (-3.0, "-3"), (np.int64(7), "7"), (999999999999999.0, "999999999999999"), (1e15, "1e+15"),
    (2e20, "2e+20"), (0.30000000000000004, "0.30000000000000004"),
])
def test_format_number_matches_gecko(value, text):
    assert format_number(value) == text


def test_kcats_from_dlkcat_and_okp_write_files(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    data = tmp_path / "data"
    data.mkdir()
    (data / "DLKcat.tsv").write_text("placeholder\n", encoding="utf-8")
    dlkcat = _kcat_list(full).assign(source="DLKcat", wildcard_level=np.nan, origin=np.nan)
    monkeypatch.setattr(kcats_from_module, "read_dlkcat_output", lambda model, path: dlkcat)
    kcats_from_dlkcat(full)
    records, meta = read_kcat_source(data / "kcats" / "dlkcat.tsv")
    assert meta["generator"].startswith("kcats_from_dlkcat")
    assert set(records["source"]) == {"DLKcat"}

    # A result stored by fetch_open_kinetics_predictor alone is taken over
    # when its job is for the method and its predictions are of that method.
    (data / "OKP_job.txt").write_text("jobId: abc123\nmethod: CataPro\n", encoding="utf-8")
    (data / "OKP_output.csv").write_text("placeholder\n", encoding="utf-8")
    okp = _kcat_list(full).assign(source="OKP-CataPro", wildcard_level=np.nan, origin=np.nan)
    monkeypatch.setattr(kcats_from_module, "parse_okp_output", lambda model, path: okp)
    kcat_list, done = kcats_from_okp(full, method="CataPro", use_stored=True)
    assert done
    assert (data / "OKP_output_catapro.csv").is_file()
    records, meta = read_kcat_source(data / "kcats" / "okp_catapro.tsv")
    assert meta["data"] == "OKP job abc123"
    assert meta["parameters"] == '{"method":"CataPro"}'
    # The shared job names UniKP, but the shared result holds CataPro
    # predictions: refused, and nothing is written.
    (data / "OKP_job.txt").write_text("jobId: uni1\nmethod: UniKP\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        kcats_from_okp(full, method="UniKP", use_stored=True)
    assert not (data / "kcats" / "okp_unikp.tsv").exists()
    with pytest.raises(ValueError):
        kcats_from_dlkcat(full, ec_rxns=["R2"])
    with pytest.raises(ValueError):
        kcats_from_okp(full, ec_rxns=[])


def test_okp_jobs_per_method(tmp_path, monkeypatch):
    """The sequence that mixed up methods: two jobs in flight, one fetched,
    then use_stored for the other. Each method keeps its own job and result."""
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    data = tmp_path / "data"
    data.mkdir()
    results = {"job-CataPro": "OKP-CataPro", "job-UniKP": "OKP-UniKP"}
    finished = set()
    calls = []

    def fake_submit(model, **kw):
        calls.append(("submit", kw["method"], kw.get("ec_rxns")))
        (data / "OKP_job.txt").write_text(f"jobId: job-{kw['method']}\nmethod: {kw['method']}\n", encoding="utf-8")
        return f"job-{kw['method']}"

    def fake_fetch(model, **kw):
        calls.append(("fetch", kw["job_id"]))
        if kw["job_id"] not in finished:
            return False, None
        Path(kw["out_file"]).write_text(results[kw["job_id"]], encoding="utf-8")
        (data / "OKP_output.csv").write_text(results[kw["job_id"]], encoding="utf-8")
        return True, _kcat_list(model).assign(source=results[kw["job_id"]], wildcard_level=np.nan, origin=np.nan)

    def fake_parse(model, path):
        return _kcat_list(model).assign(source=Path(path).read_text(), wildcard_level=np.nan, origin=np.nan)

    monkeypatch.setattr(kcats_from_module, "submit_open_kinetics_predictor", fake_submit)
    monkeypatch.setattr(kcats_from_module, "fetch_open_kinetics_predictor", fake_fetch)
    monkeypatch.setattr(kcats_from_module, "parse_okp_output", fake_parse)

    assert kcats_from_okp(full, method="CataPro") == (None, False)
    assert kcats_from_okp(full, method="UniKP", ec_rxns=["R2"]) == (None, False)
    finished.add("job-CataPro")
    assert kcats_from_okp(full, method="CataPro")[1]
    with pytest.raises(FileNotFoundError):
        kcats_from_okp(full, method="UniKP", use_stored=True)
    assert not (data / "kcats" / "okp_unikp.tsv").exists()
    records, _ = read_kcat_source(data / "kcats" / "okp_catapro.tsv")
    assert set(records["source"]) == {"OKP-CataPro"}

    finished.add("job-UniKP")
    kcats_from_okp(full, method="UniKP")
    records, meta = read_kcat_source(data / "kcats" / "okp_unikp.tsv")
    assert set(records["source"]) == {"OKP-UniKP"}
    assert meta["parameters"] == '{"ec_rxns":1,"method":"UniKP"}'
    assert meta["data"] == "OKP job job-UniKP"

    # ec_rxns without a new job is an error; a resubmit removes the old result.
    with pytest.raises(ValueError):
        kcats_from_okp(full, method="UniKP", ec_rxns=["R2"])
    finished.discard("job-UniKP")
    assert kcats_from_okp(full, method="UniKP", resubmit=True) == (None, False)
    assert not (data / "OKP_output_unikp.csv").exists()
    with pytest.raises(FileNotFoundError):
        kcats_from_okp(full, method="UniKP", use_stored=True)
    assert calls[0] == ("submit", "CataPro", None) and calls[2] == ("submit", "UniKP", ["R2"])


def test_kcats_from_brenda_writes_file(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    seen = {}

    def fake_fuzzy(model, brenda, phyl, **kw):
        seen.update(kw)
        return _kcat_list(model)

    monkeypatch.setattr(kcats_from_module, "fuzzy_kcat_matching", fake_fuzzy)
    kcats_from_brenda(full, aggregation="Median", brenda=object(), phyl_dist=object())
    assert seen["adapter"] is adapter and seen["aggregate"] == "median"
    records, meta = read_kcat_source(tmp_path / "data" / "kcats" / "brenda.tsv")
    assert meta["parameters"] == '{"aggregation":"median","force_wildcard_level":0}'
    assert len(records) == len(full.ec.rxns)


def test_parameters_json_matches_gecko():
    """The same text as GECKO's testGuardsNumbersAndJson_tc0078."""
    from geckopy.gather_kcats.kcat_sources import _parameters_json

    value = {"z": 1.0, "a": {"y": "café", "b": True}, "n": np.nan, "f": 1 / 3, "g": 1e-7,
             "h": 1e15, "k": 1e20, "l": [5], "s": 'tab\tquote"'}
    assert _parameters_json(value) == (
        '{"a":{"b":true,"y":"café"},"f":0.3333333333333333,"g":1e-07,"h":1e+15,'
        '"k":1e+20,"l":[5],"n":null,"s":"tab\\tquote\\"","z":1}'
    )
    assert _parameters_json({"x": np.int64(3), "y": np.array([1.5, 2.0]), "i": np.inf}) == \
        '{"i":null,"x":3,"y":[1.5,2]}'


def test_enzymeless_rows_are_not_written(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    mat = full.ec.rxn_enz_mat.tolil()
    i = full.ec.rxns.index("R1_EXP_2")
    mat[i, :] = 0
    full.ec.rxn_enz_mat = mat.tocsr()
    save_kcat_source(full, _kcat_list(full), "brenda")
    loaded, _ = load_kcat_source(full, "brenda")
    assert "R1_EXP_2" not in set(loaded["rxn_id"])
    assert (loaded["rxn_id"] == "R1_EXP_1").sum() == 1


def test_rows_are_sorted(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    kl = _kcat_list(full).iloc[::-1]
    path = save_kcat_source(full, kl, "brenda")
    records, _ = read_kcat_source(path)
    keys = list(zip(records["rxn"], records["enzymes"], records["substrate"], records["source"]))
    assert keys == sorted(keys)
    assert b"\r\n" not in path.read_bytes()


def test_shared_fixture():
    """The fixture file, read the same way by GECKO: byte-order mark, CRLF,
    a quoted header line and field, comments and blank lines between rows,
    a row of tabs, a decimal comma, a negative kcat."""
    path = Path(__file__).resolve().parent / "data" / "kcatSourceFixture.tsv"
    records, meta = read_kcat_source(path)
    assert meta["source"] == "fixture"
    assert meta["parameters"] == '{"aggregation":"median"}'
    assert meta["kind"] == "curated"
    assert list(records["line"]) == [7, 9, 11, 12, 13, 14, 15]
    assert list(records["rxn"]) == ["R2", "R3", "R5", "R2_REV", "", "R9", "R9"]
    assert records["substrate"].iloc[1] == "2,3-bisphospho-D-glycerate"
    kcats = records["kcat"].tolist()
    assert kcats[:2] == [12.5, 1e-05] and np.isnan(kcats[2]) and kcats[3:] == [7, 3, -1, 4]
    assert records["log10_sd"].iloc[0] == 0.3
    assert records["kcat"].dtype == float


def test_extra_fields_are_an_error(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("rxn\tkcat\tsource\nR2\t1\tbrenda\textra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more fields"):
        read_kcat_source(path)


def test_guards_numbers_and_parity_of_reading(tmp_path):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    folder = tmp_path / "data" / "kcats"
    folder.mkdir(parents=True)
    # The curated guard reads quoted header lines and ignores the case of keys.
    (folder / "a.tsv").write_text('"# kind: curated"\nrxn\tkcat\tsource\n', encoding="utf-8")
    (folder / "b.tsv").write_text("# Kind: Curated\nrxn\tkcat\tsource\n", encoding="utf-8")
    (folder / "c.tsv").write_text("# SOURCE: custom\nrxn\tkcat\tsource\n", encoding="utf-8")
    for name in ("a", "b", "c"):
        with pytest.raises(PermissionError):
            save_kcat_source(full, _kcat_list(full), name)
    # The source functions check the file before they gather.
    with pytest.raises(PermissionError):
        kcats_from_brenda(full, save_as="custom", brenda=object(), phyl_dist=object())

    # Infinite, NaN and non-positive kcats are not written, and not applied.
    kl = _kcat_list(full)
    kl.loc[0, "kcat"], kl.loc[1, "kcat"], kl.loc[2, "kcat"] = np.inf, np.nan, -2.0
    path = save_kcat_source(full, kl, "numbers", data="two\nlines")
    records, meta = read_kcat_source(path)
    assert (records["kcat"] > 0).all() and np.isfinite(records["kcat"]).all()
    assert meta["data"] == "two lines"
    (folder / "inf.tsv").write_text("rxn\tenzymes\tkcat\tsource\nR2\tP4\tInf\tcustom\nR2\tP4\t1e999\tcustom\n"
                                    "R2\tP4\t\u0661\u0662\tcustom\nR2\tP4\t\" 12 \"\tcustom\n", encoding="utf-8")
    loaded, report = load_kcat_source(full, folder / "inf.tsv")
    assert list(report["invalid"]["line"]) == [2, 3, 4]
    assert list(loaded["kcat"]) == [12.0]  # quoted with spaces: trimmed, as in GECKO

    # Substrates come back as lists; whole-number kcats as float.
    loaded, _ = load_kcat_source(full, path)
    assert loaded["substrates"].iloc[0] == ["m1", "ATP"]
    assert loaded["kcat"].dtype == float

    # A column name twice, or a column named line, is an error.
    (folder / "dup.tsv").write_text("rxn\tkcat\tkcat\tsource\n", encoding="utf-8")
    (folder / "line.tsv").write_text("rxn\tkcat\tsource\tline\n", encoding="utf-8")
    for name in ("dup", "line"):
        with pytest.raises(ValueError):
            read_kcat_source(folder / f"{name}.tsv")


def test_rows_sorted_by_every_column(tmp_path):
    """The same order as GECKO's testGuardsNumbersAndJson_tc0078."""
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    folder = tmp_path / "data" / "kcats"
    folder.mkdir(parents=True)
    (folder / "unsorted.tsv").write_text(
        "rxn\tenzymes\tsubstrate\tkcat\tsource\tdetail\n"
        "R2\tP4\tm1\t2\tb\t\nR2\tP4\tm1\t2\ta\twc=1\nR2\tP4\t\t5\ta\t\nR1_REV\tP3\tm2\t1\ta\t\n", encoding="utf-8")
    loaded, _ = load_kcat_source(full, folder / "unsorted.tsv")
    path = save_kcat_source(full, loaded, "sorted")
    records, _ = read_kcat_source(path)
    assert list(records["rxn"]) == ["R1_REV", "R2", "R2", "R2"]
    assert list(records["substrate"]) == ["m2", "", "m1", "m1"]
    assert list(records["source"]) == ["a", "a", "a", "b"]


def test_subset_warns(tmp_path, monkeypatch, caplog):
    adapter = _adapter(tmp_path)
    full = _model(adapter, light=False)
    monkeypatch.setattr(kcats_from_module, "fuzzy_kcat_matching", lambda model, b, p, **kw: _kcat_list(model))
    kcats_from_brenda(full, brenda=object(), phyl_dist=object())
    with caplog.at_level("WARNING"):
        kcats_from_brenda(full, ec_rxns=["R2"], brenda=object(), phyl_dist=object())
    assert "replaced by kcats for 1 reactions only" in caplog.text
