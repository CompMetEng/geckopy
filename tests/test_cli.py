"""Tests for the template generator and the CLI scaffold."""
import tomllib
from pathlib import Path

from geckopy import ModelAdapter
from geckopy.adapter.template import generate_template_toml
from geckopy.cli import main


def test_generated_template_parses_as_toml():
    text = generate_template_toml()
    parsed = tomllib.loads(text)
    # Required fields are uncommented, so they parse into the dict.
    assert "conv_gem" in parsed
    assert "org_name" in parsed
    # Optional fields are commented out, so they do not appear.
    assert "sigma" not in parsed


def test_template_includes_nested_sections():
    text = generate_template_toml()
    assert "[kegg]" in text
    assert "[uniprot]" in text
    assert "[complex]" in text
    assert "[evotune]" in text


def test_template_renders_dict_valued_evotune_fields():
    """The evotune section carries per-source tables, not just scalars."""
    text = generate_template_toml()
    uncommented = "\n".join(
        line.lstrip("# ") for line in text.splitlines()
        if line.lstrip("# ").startswith(("sigma0_log_source", "source_groups"))
    )
    parsed = tomllib.loads(uncommented)
    assert parsed["sigma0_log_source"]["brenda"] == 0.2
    assert parsed["source_groups"]["brenda"]["sources"] == ["brenda"]


def test_template_includes_field_descriptions_as_comments():
    text = generate_template_toml()
    assert "saturation" in text.lower()


def test_cli_init_creates_folder_structure(tmp_path):
    target = tmp_path / "my-ec"
    exit_code = main(["init", str(target)])
    assert exit_code == 0
    assert (target / "model_adapter.toml").is_file()
    assert (target / "adapter.py").is_file()
    assert (target / "data").is_dir()
    assert (target / "models").is_dir()
    assert (target / "output").is_dir()
    assert (target / ".gitignore").read_text().strip() == "data/okpApiKey.txt"


def test_cli_init_refuses_nonempty_folder(tmp_path, capsys):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "something.txt").write_text("hello")

    exit_code = main(["init", str(target)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not empty" in captured.err


def test_cli_init_scaffolds_the_evotune_section(tmp_path):
    target = tmp_path / "my-ec"
    main(["init", str(target)])
    toml_text = (target / "model_adapter.toml").read_text()
    assert "[evotune]" in toml_text


def test_cli_init_produces_loadable_adapter_after_filling_required(tmp_path):
    """End-to-end: scaffold, fill in required fields, load via ModelAdapter."""
    target = tmp_path / "my-ec"
    main(["init", str(target)])

    toml_path = target / "model_adapter.toml"
    toml_text = toml_path.read_text()
    toml_text = toml_text.replace(
        '"models/REPLACE_ME.xml"', '"models/real.xml"'
    ).replace('"REPLACE_ME"', '"Testus testus"')
    toml_path.write_text(toml_text)

    adapter = ModelAdapter.from_folder(target)
    assert adapter.params.org_name == "Testus testus"
    # Defaults kick in for fields the user did not uncomment.
    assert adapter.params.sigma == 0.5
    assert adapter.params.kegg.id == "sce"


def test_cli_init_stub_adapter_py_is_commented_out(tmp_path):
    """Generated adapter.py should define nothing at import time."""
    target = tmp_path / "my-ec"
    main(["init", str(target)])
    namespace: dict = {}
    exec((target / "adapter.py").read_text(), namespace)
    user_classes = [
        v for k, v in namespace.items()
        if not k.startswith("_") and isinstance(v, type)
    ]
    assert user_classes == []


def test_cli_brenda_refresh_empty_cache_errors(tmp_path, capsys):
    rc = main([
        "brenda-refresh",
        "--cache-dir", str(tmp_path / "cache"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "download.php" in err


def test_cli_brenda_refresh_end_to_end(tmp_path, capsys):
    """Drop the test fixture JSON in the cache dir and run the command."""
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    fixture = (
        Path(__file__).parent / "data" / "brenda_minimal.json"
    ).read_text()
    (cache / "brenda_minimal.json").write_text(fixture)

    rc = main([
        "brenda-refresh",
        "--cache-dir", str(cache),
        "--out-dir", str(out),
    ])
    assert rc == 0
    output = capsys.readouterr().out
    assert "BRENDA release 2026.1" in output
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        assert (out / name).exists()


def test_cli_uniprot_download_invokes_function(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_download(uid, path, *, id_type, gene_id_field, reviewed):
        calls["args"] = (uid, str(path), id_type, gene_id_field, reviewed)
        Path(path).write_text("Entry\tGene\tEC\tMass\tSequence\n")
        return path

    monkeypatch.setattr(
        "geckopy.cli.download_uniprot", fake_download,
    )
    out = tmp_path / "u.tsv"
    rc = main(["uniprot-download", "559292", str(out)])
    assert rc == 0
    assert calls["args"][0] == "559292"
    assert calls["args"][2] == "taxonomy_id"
    assert calls["args"][3] == "gene_oln"
    assert calls["args"][4] is False
    assert out.exists()


def test_cli_uniprot_download_reviewed_only(tmp_path, monkeypatch):
    calls = {}

    def fake_download(uid, path, *, id_type, gene_id_field, reviewed):
        calls["reviewed"] = reviewed
        Path(path).write_text("")
        return path

    monkeypatch.setattr(
        "geckopy.cli.download_uniprot", fake_download,
    )
    main([
        "uniprot-download", "559292", str(tmp_path / "u.tsv"),
        "--reviewed-only",
    ])
    assert calls["reviewed"] is True


def test_cli_kegg_download_invokes_function(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_download(kid, path, *, gene_id_field):
        calls["args"] = (kid, str(path), gene_id_field)
        Path(path).write_text("P1,G1,K1,1.1.1.1,10000,p,SEQ\n")
        return path

    monkeypatch.setattr(
        "geckopy.cli.download_kegg", fake_download,
    )
    out = tmp_path / "k.tsv"
    rc = main(["kegg-download", "sce", str(out)])
    assert rc == 0
    assert calls["args"][0] == "sce"
    assert calls["args"][2] == "kegg"
    assert out.exists()
