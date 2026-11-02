"""Tests for get_complex_data (network mocked)."""

import pytest
import requests

from geckopy.databases import (
    ComplexPortalEntry,
    get_complex_data,
    load_complex_portal_json,
)


# --------------------------------------------------------------------------- #
# Fake HTTP layer
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _install_fake_session(monkeypatch, responses: dict):
    """Replace _make_session with a fake that returns canned responses.

    `responses` maps URL substrings to (status_code, payload). The first
    URL substring that matches the request URL wins.
    """
    from geckopy.databases import complex_portal_download as mod

    class _FakeSession:
        def get(self, url, params=None, timeout=None):
            for fragment, (status, payload) in responses.items():
                if fragment in url:
                    return _FakeResponse(status, payload)
            return _FakeResponse(404, {})

    def _factory():
        return _FakeSession()

    monkeypatch.setattr(mod, "_make_session", _factory)


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #

def test_returns_list_of_entries(monkeypatch):
    responses = {
        "search/*?facets": (200, {
            "size": 1, "totalNumberOfResults": 1,
            "elements": [{"complexAC": "CPX-1"}],
        }),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1",
            "name": "Test complex",
            "species": "Homo sapiens",
            "participants": [
                {"interactorType": "protein", "name": "G1",
                 "identifier": "P1", "stochiometry": "minValue: 1, maxValue: 1"},
                {"interactorType": "protein", "name": "G2",
                 "identifier": "P2", "stochiometry": "minValue: 2, maxValue: 2"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    assert len(result) == 1
    assert isinstance(result[0], ComplexPortalEntry)
    assert result[0].complex_id == "CPX-1"
    assert result[0].protein_ids == ["P1", "P2"]
    assert result[0].stoichiometry == [1, 2]


def test_taxonomic_id_zero_queries_all(monkeypatch):
    """taxonomic_id=0 means no species filter; just builds the bare URL."""
    captured_urls: list[str] = []

    class _CaptureSession:
        def get(self, url, params=None, timeout=None):
            captured_urls.append(url)
            if "search/*" in url and "filters" not in url:
                return _FakeResponse(200, {"size": 0, "totalNumberOfResults": 0, "elements": []})
            return _FakeResponse(404, {})

    from geckopy.databases import complex_portal_download as mod
    monkeypatch.setattr(mod, "_make_session", lambda: _CaptureSession())

    with pytest.raises(ValueError, match="No complexes"):
        get_complex_data(taxonomic_id=0)

    assert any(u.endswith("search/*") for u in captured_urls)


def test_none_taxonomic_id_raises():
    with pytest.raises(ValueError, match="taxonomic_id is required"):
        get_complex_data(taxonomic_id=None)


def test_empty_search_response_raises(monkeypatch):
    _install_fake_session(monkeypatch, {
        "search/*": (200, {"size": 0, "totalNumberOfResults": 0, "elements": []}),
    })
    with pytest.raises(ValueError, match="No complexes"):
        get_complex_data(taxonomic_id=9606)


def test_404_on_individual_complex_is_skipped(monkeypatch):
    """If one complex's details 404, the run continues without it."""
    responses = {
        "search/*": (200, {
            "size": 2, "totalNumberOfResults": 2,
            "elements": [{"complexAC": "CPX-1"}, {"complexAC": "CPX-MISSING"}],
        }),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1",
            "name": "Test",
            "species": "x",
            "participants": [
                {"interactorType": "protein", "name": "G1",
                 "identifier": "P1", "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
        "complex/CPX-MISSING": (404, {}),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    assert len(result) == 1
    assert result[0].complex_id == "CPX-1"


# --------------------------------------------------------------------------- #
# Stoichiometry parsing
# --------------------------------------------------------------------------- #

def test_missing_stoichiometry_becomes_zero(monkeypatch):
    """A complex with no stoichiometry info should have all zeros."""
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "CPX-1"}]}),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1",
            "name": "x", "species": "y",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1"},
                {"interactorType": "protein", "name": "G2", "identifier": "P2"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    assert result[0].stoichiometry == [0, 0]


def test_only_min_value_parsed(monkeypatch):
    """Only minValue is parsed from a 'minValue: 2, maxValue: 4' string;
    maxValue is ignored."""
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "CPX-1"}]}),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1", "name": "x", "species": "y",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 2, maxValue: 4"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)
    result = get_complex_data(taxonomic_id=9606)
    assert result[0].stoichiometry == [2]


# --------------------------------------------------------------------------- #
# Filter to protein participants
# --------------------------------------------------------------------------- #

def test_non_protein_participants_filtered(monkeypatch):
    """Participants with interactorType != 'protein' are excluded when
    at least one protein participant is present."""
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "CPX-1"}]}),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1", "name": "x", "species": "y",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 1, maxValue: 1"},
                {"interactorType": "small molecule", "name": "ligand",
                 "identifier": "L1",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)
    result = get_complex_data(taxonomic_id=9606)
    assert result[0].protein_ids == ["P1"]
    assert result[0].gene_names == ["G1"]


# --------------------------------------------------------------------------- #
# Write-to-disk side effect
# --------------------------------------------------------------------------- #

def test_writes_json_to_disk(monkeypatch, tmp_path):
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "CPX-1"}]}),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1", "name": "Test", "species": "Homo sapiens",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    out_path = tmp_path / "ComplexPortal.json"
    get_complex_data(taxonomic_id=9606, write_to=out_path)

    assert out_path.is_file()
    # File should be loadable via the standard loader.
    loaded = load_complex_portal_json(out_path)
    assert len(loaded) == 1
    assert loaded[0].complex_id == "CPX-1"


def test_round_trip_through_disk(monkeypatch, tmp_path):
    """Download, write, reload: the entries should be identical
    in their relevant fields."""
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "CPX-1"}]}),
        "complex/CPX-1": (200, {
            "complexAc": "CPX-1", "name": "Test", "species": "x",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 1, maxValue: 1"},
                {"interactorType": "protein", "name": "G2", "identifier": "P2",
                 "stochiometry": "minValue: 3, maxValue: 3"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    path = tmp_path / "ComplexPortal.json"
    downloaded = get_complex_data(taxonomic_id=9606, write_to=path)
    reloaded = load_complex_portal_json(path)

    assert len(downloaded) == len(reloaded) == 1
    assert downloaded[0].complex_id == reloaded[0].complex_id
    assert downloaded[0].protein_ids == reloaded[0].protein_ids
    assert downloaded[0].stoichiometry == reloaded[0].stoichiometry


# --------------------------------------------------------------------------- #
# Sub-complex flattening
# --------------------------------------------------------------------------- #

def test_complex_of_complexes_flattens(monkeypatch):
    """A complex marked defined=2 (no protein participants, only complex
    participants) should be flattened by joining its sub-complexes'
    proteins, with stoichiometries multiplied."""
    responses = {
        "search/*": (200, {
            "size": 3, "totalNumberOfResults": 3,
            "elements": [
                {"complexAC": "SUB-A"},
                {"complexAC": "SUB-B"},
                {"complexAC": "BIG"},
            ],
        }),
        "complex/SUB-A": (200, {
            "complexAc": "SUB-A", "name": "A", "species": "x",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 2, maxValue: 2"},
            ],
        }),
        "complex/SUB-B": (200, {
            "complexAc": "SUB-B", "name": "B", "species": "x",
            "participants": [
                {"interactorType": "protein", "name": "G2", "identifier": "P2",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
        "complex/BIG": (200, {
            "complexAc": "BIG", "name": "Big", "species": "x",
            "participants": [
                {"interactorType": "complex", "name": "SUB-A", "identifier": "SUB-A",
                 "stochiometry": "minValue: 3, maxValue: 3"},
                {"interactorType": "complex", "name": "SUB-B", "identifier": "SUB-B",
                 "stochiometry": "minValue: 5, maxValue: 5"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    big = next(r for r in result if r.complex_id == "BIG")
    # SUB-A contributes P1 with stoich 2*3=6.
    # SUB-B contributes P2 with stoich 1*5=5.
    assert big.protein_ids == ["P1", "P2"]
    assert big.stoichiometry == [6, 5]


def test_subcomplex_with_ligand_ignores_ligand(monkeypatch):
    """A complex with a sub-complex participant *and* a ligand (no direct
    protein) flattens to the sub-complex proteins only; the ligand must not
    be treated as a sub-complex id."""
    responses = {
        "search/*": (200, {
            "size": 2, "totalNumberOfResults": 2,
            "elements": [{"complexAC": "SUB-A"}, {"complexAC": "BIG"}],
        }),
        "complex/SUB-A": (200, {
            "complexAc": "SUB-A", "name": "A", "species": "x",
            "participants": [
                {"interactorType": "protein", "name": "G1", "identifier": "P1",
                 "stochiometry": "minValue: 2, maxValue: 2"},
            ],
        }),
        "complex/BIG": (200, {
            "complexAc": "BIG", "name": "Big", "species": "x",
            "participants": [
                {"interactorType": "complex", "name": "SUB-A",
                 "identifier": "SUB-A",
                 "stochiometry": "minValue: 3, maxValue: 3"},
                {"interactorType": "small molecule", "name": "ATP",
                 "identifier": "CHEBI:15422",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    big = next(r for r in result if r.complex_id == "BIG")
    assert big.protein_ids == ["P1"]
    assert big.stoichiometry == [6]


def test_ligand_only_complex_has_no_proteins(monkeypatch):
    """A complex whose only participants are non-protein, non-complex must
    not be flattened as a complex-of-complexes; it yields no proteins."""
    responses = {
        "search/*": (200, {"size": 1, "totalNumberOfResults": 1, "elements": [{"complexAC": "LIG"}]}),
        "complex/LIG": (200, {
            "complexAc": "LIG", "name": "ligand only", "species": "x",
            "participants": [
                {"interactorType": "small molecule", "name": "ATP",
                 "identifier": "CHEBI:15422",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }),
    }
    _install_fake_session(monkeypatch, responses)

    result = get_complex_data(taxonomic_id=9606)
    lig = next(r for r in result if r.complex_id == "LIG")
    assert lig.protein_ids == []


def test_search_paginates(monkeypatch):
    """The search is paged: complexes beyond the first page must still be
    collected. ``first`` is a record offset, and ``totalNumberOfResults``
    -- not ``size``, which counts only the page at hand -- is the total."""
    from geckopy.databases import complex_portal_download as mod

    page_size = mod._SEARCH_PAGE_SIZE
    pages = {
        0: {"size": 2, "totalNumberOfResults": 3,
            "elements": [{"complexAC": "CPX-1"}, {"complexAC": "CPX-2"}]},
        page_size: {"size": 1, "totalNumberOfResults": 3,
                    "elements": [{"complexAC": "CPX-3"}]},
    }

    def _detail(cid):
        return {
            "complexAc": cid, "name": cid, "species": "x",
            "participants": [
                {"interactorType": "protein", "name": f"G_{cid}",
                 "identifier": f"P_{cid}",
                 "stochiometry": "minValue: 1, maxValue: 1"},
            ],
        }

    class _PagingSession:
        def get(self, url, params=None, timeout=None):
            if "search/" in url:
                offset = (params or {}).get("first", 0)
                return _FakeResponse(200, pages.get(
                    offset,
                    {"size": 0, "totalNumberOfResults": 3, "elements": []},
                ))
            cid = url.rsplit("/", 1)[-1]
            return _FakeResponse(200, _detail(cid))

    monkeypatch.setattr(mod, "_make_session", lambda: _PagingSession())

    result = get_complex_data(taxonomic_id=9606)
    assert {r.complex_id for r in result} == {"CPX-1", "CPX-2", "CPX-3"}
