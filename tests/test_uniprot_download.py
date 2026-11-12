"""Tests for download_uniprot using mocked HTTP."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from geckopy.databases.uniprot_download import download_uniprot


def _make_session(response_text: str, *, status: int = 200) -> MagicMock:
    sess = MagicMock()
    captured = {"params": None, "url": None}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        resp = MagicMock()
        resp.status_code = status
        resp.text = response_text
        if status >= 400:
            from requests import HTTPError
            resp.raise_for_status = MagicMock(side_effect=HTTPError(f"HTTP {status}"))
        else:
            resp.raise_for_status = MagicMock()
        return resp

    sess.get.side_effect = fake_get
    sess._captured = captured
    return sess


def test_basic_download_writes_response_verbatim(tmp_path):
    payload = (
        "Entry\tGene Names (ordered locus)\tEC number\tMass\tSequence\n"
        "P00330\tYOL086C\t1.1.1.1\t36850\tMSIPETQK\n"
    )
    sess = _make_session(payload)
    out = tmp_path / "uniprot.tsv"
    download_uniprot(559292, out, session=sess)
    assert out.read_text(encoding="utf-8") == payload


def test_query_includes_reviewed_filter_when_true(tmp_path):
    sess = _make_session("Entry\tGene\tEC\tMass\tSequence\nP1\tG1\t\t10\tA\n")
    download_uniprot(559292, tmp_path / "u.tsv", reviewed=True, session=sess)
    params = sess._captured["params"]
    assert "reviewed:true" in params["query"]
    assert "taxonomy_id:559292" in params["query"]


def test_query_has_no_reviewed_filter_by_default(tmp_path):
    """The default follows the adapter default uniprot.reviewed = false."""
    sess = _make_session("Entry\tGene\tEC\tMass\tSequence\nP1\tG1\t\t10\tA\n")
    download_uniprot(559292, tmp_path / "u.tsv", session=sess)
    assert "reviewed" not in sess._captured["params"]["query"]


def test_taxonomy_alias_normalised_to_taxonomy_id(tmp_path):
    sess = _make_session("Entry\tGene\tEC\tMass\tSequence\nP1\tG1\t\t10\tA\n")
    download_uniprot(
        559292, tmp_path / "u.tsv", id_type="taxonomy", session=sess,
    )
    assert "taxonomy_id:559292" in sess._captured["params"]["query"]


def test_gene_id_field_threaded_into_fields_param(tmp_path):
    sess = _make_session("Entry\tGene\tEC\tMass\tSequence\nP1\tG1\t\t10\tA\n")
    download_uniprot(
        559292, tmp_path / "u.tsv", gene_id_field="gene_primary", session=sess,
    )
    assert "gene_primary" in sess._captured["params"]["fields"]


def test_empty_response_raises(tmp_path):
    sess = _make_session("")
    with pytest.raises(RuntimeError, match="empty response"):
        download_uniprot(999999, tmp_path / "u.tsv", session=sess)


def test_http_error_after_retries_raises(tmp_path):
    sess = _make_session("ignored", status=500)
    with pytest.raises(RuntimeError, match="failed after"):
        download_uniprot(559292, tmp_path / "u.tsv", session=sess)


def test_creates_parent_dirs(tmp_path):
    payload = "Entry\tGene\tEC\tMass\tSequence\nP1\tG1\t\t10\tA\n"
    sess = _make_session(payload)
    target = tmp_path / "nested" / "subdir" / "uniprot.tsv"
    download_uniprot(559292, target, session=sess)
    assert target.is_file()


def test_round_trip_with_load_uniprot_tsv(tmp_path):
    """download_uniprot output is consumable by load_uniprot_tsv."""
    from geckopy.databases import load_uniprot_tsv
    payload = (
        "Entry\tGene Names (ordered locus)\tEC number\tMass\tSequence\n"
        "P00330\tYOL086C\t1.1.1.1\t36850\tMSIPETQK\n"
        "P00331\tYDL168W\t1.1.1.1\t36755\tMSAPTQAE\n"
    )
    sess = _make_session(payload)
    out = tmp_path / "uniprot.tsv"
    download_uniprot(559292, out, session=sess)
    db = load_uniprot_tsv(out)
    assert len(db) == 2
    assert db.ids == ["P00330", "P00331"]
    assert db.genes == ["YOL086C", "YDL168W"]
