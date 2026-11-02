"""Tests for load_phyl_dist."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io import savemat

from geckopy.databases import PhylDist, download_phyl_dist, load_phyl_dist
from geckopy.databases.phyl_dist import _clean_name


# --------------------------------------------------------------------------- #
# Fixture builder
# --------------------------------------------------------------------------- #

def _write_phyl_dist_mat(
    path: Path,
    names: list[str],
    dist_mat: np.ndarray,
) -> None:
    """Write a small PhylDist.mat fixture in the format the loader expects."""
    struct = {
        "names": np.array(names, dtype=object),
        "distMat": np.asarray(dist_mat, dtype=float),
        "ids": np.array([f"id{i}" for i in range(len(names))], dtype=object),
    }
    savemat(str(path), {"phylDistStruct": struct})


# --------------------------------------------------------------------------- #
# Missing / malformed files
# --------------------------------------------------------------------------- #

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="PhylDist.mat not found"):
        load_phyl_dist(tmp_path / "nonexistent.mat")


def test_mat_without_phyl_dist_struct_raises(tmp_path):
    p = tmp_path / "wrong.mat"
    savemat(str(p), {"some_other_var": np.zeros(3)})
    with pytest.raises(KeyError, match="expected variable 'phylDistStruct'"):
        load_phyl_dist(p)


def test_phyl_dist_struct_missing_field_raises(tmp_path):
    p = tmp_path / "incomplete.mat"
    savemat(
        str(p),
        {"phylDistStruct": {"names": np.array(["a", "b"], dtype=object)}},
    )
    with pytest.raises(KeyError, match="missing 'names' or 'distMat'"):
        load_phyl_dist(p)


# --------------------------------------------------------------------------- #
# Basic loading
# --------------------------------------------------------------------------- #

def test_basic_load_two_organisms(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["Saccharomyces cerevisiae", "Escherichia coli"],
        dist_mat=np.array([[0.0, 1.5], [1.5, 0.0]]),
    )

    pd = load_phyl_dist(p)

    assert isinstance(pd, PhylDist)
    assert pd.names == ["Saccharomyces cerevisiae", "Escherichia coli"]
    assert pd.dist_matrix.shape == (2, 2)
    assert pd.dist_matrix[0, 1] == pytest.approx(1.5)


def test_returns_phyl_dist_dataclass(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(p, ["foo"], np.array([[0.0]]))
    pd = load_phyl_dist(p)
    assert isinstance(pd, PhylDist)
    # All four documented attributes present.
    assert pd.names is not None
    assert pd.dist_matrix is not None
    assert pd.name_to_index is not None
    assert pd.genus_to_indices is not None


# --------------------------------------------------------------------------- #
# Name cleaning (parenthetical stripping)
# --------------------------------------------------------------------------- #

def test_clean_name_strips_trailing_parenthetical():
    assert _clean_name("Saccharomyces cerevisiae (baker's yeast)") == (
        "Saccharomyces cerevisiae"
    )


def test_clean_name_no_parenthetical_unchanged():
    assert _clean_name("Escherichia coli") == "Escherichia coli"


def test_clean_name_strips_internal_paren_to_end():
    """Strips everything from the first '(' onward, even when more
    parenthetical groups follow."""
    assert _clean_name("foo (bar) (baz)") == "foo"


def test_clean_name_strips_trailing_whitespace():
    assert _clean_name("foo   ") == "foo"


def test_load_strips_parenthetical_from_names(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=[
            "Saccharomyces cerevisiae (baker's yeast)",
            "Escherichia coli (K-12)",
        ],
        dist_mat=np.zeros((2, 2)),
    )
    pd = load_phyl_dist(p)
    assert pd.names == [
        "Saccharomyces cerevisiae",
        "Escherichia coli",
    ]


# --------------------------------------------------------------------------- #
# name_to_index
# --------------------------------------------------------------------------- #

def test_name_to_index_is_lowercased(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["Saccharomyces cerevisiae", "Escherichia coli"],
        dist_mat=np.zeros((2, 2)),
    )
    pd = load_phyl_dist(p)
    assert pd.name_to_index["saccharomyces cerevisiae"] == 0
    assert pd.name_to_index["escherichia coli"] == 1


def test_name_to_index_first_occurrence_wins_for_duplicates(tmp_path):
    """When a name appears more than once, name_to_index resolves to
    the first matching row."""
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["Foo bar", "Foo bar", "Baz qux"],
        dist_mat=np.zeros((3, 3)),
    )
    pd = load_phyl_dist(p)
    assert pd.name_to_index["foo bar"] == 0


# --------------------------------------------------------------------------- #
# genus_to_indices
# --------------------------------------------------------------------------- #

def test_genus_to_indices_groups_same_genus(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=[
            "Saccharomyces cerevisiae",
            "Saccharomyces pombe",
            "Escherichia coli",
        ],
        dist_mat=np.zeros((3, 3)),
    )
    pd = load_phyl_dist(p)
    assert pd.genus_to_indices["saccharomyces"] == [0, 1]
    assert pd.genus_to_indices["escherichia"] == [2]


def test_genus_extracted_case_insensitively(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["FOO bar", "foo qux"],
        dist_mat=np.zeros((2, 2)),
    )
    pd = load_phyl_dist(p)
    assert sorted(pd.genus_to_indices["foo"]) == [0, 1]


def test_single_word_name_used_as_genus(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["ecoli"],
        dist_mat=np.zeros((1, 1)),
    )
    pd = load_phyl_dist(p)
    assert pd.genus_to_indices["ecoli"] == [0]


def test_empty_name_no_genus_entry(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["", "foo bar"],
        dist_mat=np.zeros((2, 2)),
    )
    pd = load_phyl_dist(p)
    # No empty key in genus_to_indices.
    assert "" not in pd.genus_to_indices
    assert pd.genus_to_indices["foo"] == [1]


# --------------------------------------------------------------------------- #
# Distance matrix
# --------------------------------------------------------------------------- #

def test_distance_matrix_preserved(tmp_path):
    p = tmp_path / "PhylDist.mat"
    expected = np.array([
        [0.0, 1.5, 2.7],
        [1.5, 0.0, 3.1],
        [2.7, 3.1, 0.0],
    ])
    _write_phyl_dist_mat(p, names=["a", "b", "c"], dist_mat=expected)
    pd = load_phyl_dist(p)
    np.testing.assert_array_equal(pd.dist_matrix, expected)


def test_distance_matrix_dtype_is_float(tmp_path):
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=["a", "b"],
        dist_mat=np.array([[0, 1], [1, 0]], dtype=int),
    )
    pd = load_phyl_dist(p)
    assert pd.dist_matrix.dtype == np.float64


# --------------------------------------------------------------------------- #
# Combined: name and genus lookup against a small synthetic struct
# --------------------------------------------------------------------------- #

def test_combined_lookup_for_name_and_genus_fallback(tmp_path):
    """Simulates the use case in fuzzy_kcat_matching: try direct
    name lookup, fall back to genus lookup."""
    p = tmp_path / "PhylDist.mat"
    _write_phyl_dist_mat(
        p,
        names=[
            "Saccharomyces cerevisiae",
            "Saccharomyces pombe",
            "Escherichia coli",
        ],
        dist_mat=np.array([
            [0.0, 0.5, 5.0],
            [0.5, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ]),
    )
    pd = load_phyl_dist(p)

    # Direct name match.
    assert pd.name_to_index.get("saccharomyces cerevisiae") == 0
    # Direct miss, but genus match falls back to the genus list.
    assert pd.name_to_index.get("saccharomyces unknown") is None
    assert pd.genus_to_indices.get("saccharomyces") == [0, 1]


def _session_returning(content: bytes) -> MagicMock:
    session = MagicMock()
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp
    return session


def test_download_phyl_dist_writes_loadable_file(tmp_path):
    src = tmp_path / "src.mat"
    savemat(src, {"phylDistStruct": {
        "names": np.array(["Saccharomyces cerevisiae", "Homo sapiens"], dtype=object),
        "distMat": np.array([[0.0, 3.0], [3.0, 0.0]]),
    }})
    out = tmp_path / "data" / "PhylDist.mat"
    session = _session_returning(src.read_bytes())

    assert download_phyl_dist(out, url="https://example.org/x.mat", session=session) == out
    session.get.assert_called_once()
    assert session.get.call_args.args[0] == "https://example.org/x.mat"
    assert load_phyl_dist(out).names == ["Saccharomyces cerevisiae", "Homo sapiens"]
    assert not (tmp_path / "data" / "PhylDist.mat.part").exists()


def test_download_phyl_dist_rejects_non_phyl_dist_file(tmp_path):
    src = tmp_path / "src.mat"
    savemat(src, {"other": np.zeros(2)})
    out = tmp_path / "PhylDist.mat"
    out.write_bytes(b"existing")

    with pytest.raises(KeyError, match="phylDistStruct"):
        download_phyl_dist(out, session=_session_returning(src.read_bytes()))
    assert out.read_bytes() == b"existing"
    assert not (tmp_path / "PhylDist.mat.part").exists()

