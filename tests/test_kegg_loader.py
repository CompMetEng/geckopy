"""Tests for load_kegg_tsv."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from geckopy.databases.kegg_loader import KeggDB, load_kegg_tsv


def _write_kegg(folder: Path, content: str) -> Path:
    p = folder / "kegg.tsv"
    p.write_text(content, encoding="utf-8")
    return p


def test_basic_parse(tmp_path):
    p = _write_kegg(
        tmp_path,
        "P00401,Q0045,Q0045,7.1.1.9,58798,sce00190 Oxidative phosphorylation,MVQR\n",
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert isinstance(db, KeggDB)
    assert len(db) == 1
    assert db.uniprot_ids == ["P00401"]
    assert db.genes == ["Q0045"]
    assert db.kegg_genes == ["Q0045"]
    assert db.eccodes == ["7.1.1.9"]
    assert db.mw[0] == 58798.0
    assert db.sequences == ["MVQR"]


def test_multi_row_alignment(tmp_path):
    p = _write_kegg(
        tmp_path,
        "P1,G1,K1,1.1.1.1,10000,p1,SEQ1\n"
        "P2,G2,K2,2.2.2.2,20000,p2,SEQ2\n"
        "P3,G3,K3,3.3.3.3,30000,p3,SEQ3\n",
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert len(db) == 3
    assert db.genes == ["G1", "G2", "G3"]
    assert list(db.mw) == [10000.0, 20000.0, 30000.0]


def test_empty_ec_cell(tmp_path):
    p = _write_kegg(tmp_path, "P1,G1,K1,,12345,p1,SEQ\n")
    db = load_kegg_tsv(p, auto_download=False)
    assert db.eccodes == [""]
    assert db.mw[0] == 12345.0


def test_empty_uniprot_cell(tmp_path):
    p = _write_kegg(tmp_path, ",G1,K1,1.1.1.1,12345,p1,SEQ\n")
    db = load_kegg_tsv(p, auto_download=False)
    assert db.uniprot_ids == [""]
    assert db.genes == ["G1"]


def test_quoted_pathway_with_comma(tmp_path):
    p = _write_kegg(
        tmp_path,
        'P1,G1,K1,1.4.1.4,49627,"sce00250 Alanine, aspartate metabolism",SEQ\n',
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert db.pathways == ["sce00250 Alanine, aspartate metabolism"]
    assert db.sequences == ["SEQ"]


def test_quoted_empty_pathway(tmp_path):
    p = _write_kegg(tmp_path, 'P1,G1,K1,1.1.1.1,12345,"",SEQ\n')
    db = load_kegg_tsv(p, auto_download=False)
    assert db.pathways == [""]


def test_multiple_ec_codes_in_one_cell(tmp_path):
    p = _write_kegg(
        tmp_path,
        "P1,G1,K1,1.1.1.4 1.1.1.- 1.1.1.303,46098,p,SEQ\n",
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert db.eccodes == ["1.1.1.4 1.1.1.- 1.1.1.303"]


def test_non_numeric_mw_becomes_nan(tmp_path, caplog):
    import logging
    p = _write_kegg(tmp_path, "P1,G1,K1,1.1.1.1,not-a-number,p,SEQ\n")
    with caplog.at_level(logging.WARNING):
        db = load_kegg_tsv(p, auto_download=False)
    assert math.isnan(db.mw[0])
    assert "not numeric" in caplog.text


def test_blank_rows_skipped(tmp_path):
    p = _write_kegg(
        tmp_path,
        "P1,G1,K1,1.1.1.1,10000,p,SEQ1\n"
        "\n"
        "P2,G2,K2,2.2.2.2,20000,p,SEQ2\n",
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert len(db) == 2


def test_find_by_gene(tmp_path):
    p = _write_kegg(
        tmp_path,
        "P1,G1,K1,1.1.1.1,10000,p,SEQ1\n"
        "P2,G2,K2,2.2.2.2,20000,p,SEQ2\n",
    )
    db = load_kegg_tsv(p, auto_download=False)
    assert db.find_by_gene("G2") == 1
    assert db.find_by_gene("missing") is None


def test_missing_file_no_kegg_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="kegg.tsv not found"):
        load_kegg_tsv(tmp_path / "nope.tsv", auto_download=False)


def test_missing_file_auto_download_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_kegg_tsv(
            tmp_path / "nope.tsv", kegg_id="sce", auto_download=False
        )


def test_mw_is_numpy_array(tmp_path):
    p = _write_kegg(tmp_path, "P1,G1,K1,1.1.1.1,10000,p,SEQ\n")
    db = load_kegg_tsv(p, auto_download=False)
    assert isinstance(db.mw, np.ndarray)
    assert db.mw.dtype == np.float64


def test_real_fixture_loads(tmp_path):
    """Load the tutorial's real kegg.tsv to catch parser regressions."""
    fixture = Path(__file__).resolve().parent.parent / "tutorials" / "full_ecModel" / "data" / "kegg.tsv"
    if not fixture.is_file():
        pytest.skip("tutorial kegg.tsv not present")
    db = load_kegg_tsv(fixture, auto_download=False)
    assert len(db) > 5000
    # First row from the real file: P00401,Q0045,Q0045,7.1.1.9,58798,...
    assert db.uniprot_ids[0] == "P00401"
    assert db.genes[0] == "Q0045"
    assert db.eccodes[0] == "7.1.1.9"
    assert db.mw[0] == 58798.0


def test_empty_kegg_id_skips_download(tmp_path, monkeypatch, caplog):
    """As loadDatabases without a kegg.ID: a warning, no download, no data."""
    import geckopy.databases.kegg_download as kegg_download

    monkeypatch.setattr(kegg_download, "download_kegg", lambda *a, **k: pytest.fail("downloaded"))
    with caplog.at_level("WARNING"):
        assert load_kegg_tsv(tmp_path / "kegg.tsv", kegg_id="") is None
    assert "No KEGG organism code" in caplog.text
