"""Tests for the UniProt TSV loader."""
from pathlib import Path

import numpy as np
import pytest

from geckopy.databases import UniprotDB, load_uniprot_tsv

EXAMPLE_TSV = (
    Path(__file__).parents[1] / "examples" / "ecTestGEM" / "data" / "uniprot.tsv"
)


HEADER = "Entry\tGene Names (ordered locus)\tEC number\tMass\tSequence\n"


def _write_tsv(path: Path, rows: list[str]) -> Path:
    path.write_text(HEADER + "".join(r + "\n" for r in rows))
    return path


# --------------------------------------------------------------------------- #
# Basic parsing
# --------------------------------------------------------------------------- #

def test_loads_ectestgem_fixture():
    """Smoke test against the real ecTestGEM uniprot.tsv fixture."""
    db = load_uniprot_tsv(EXAMPLE_TSV)
    assert len(db) == 5
    assert set(db.ids) == {"P1", "P2", "P3", "P4", "P5"}
    assert set(db.genes) == {"G1", "G2", "G3", "G4", "G5"}


def test_returns_uniprotdb_instance(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t1.1.1.1\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path)
    assert isinstance(db, UniprotDB)


def test_parses_basic_row(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t1.1.1.1\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path)

    assert db.ids == ["P1"]
    assert db.genes == ["G1"]
    assert db.eccodes == ["1.1.1.1"]
    assert db.sequences == ["MKAL"]
    np.testing.assert_array_almost_equal(db.mw, [10000.0])  # stored as Da


def test_mw_stored_as_da(tmp_path):
    """Mass is stored verbatim as Da, matching the TSV column."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t\t40000\tMKAL",
        "P2\tG2\t\t12500\tMKAL",
    ])
    db = load_uniprot_tsv(path)
    np.testing.assert_array_almost_equal(db.mw, [40000.0, 12500.0])


def test_empty_cells_become_empty_strings(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t\t10000\t",
    ])
    db = load_uniprot_tsv(path)
    assert db.eccodes == [""]
    assert db.sequences == [""]


def test_missing_mass_becomes_nan(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t1.1.1.1\t\tMKAL",
    ])
    db = load_uniprot_tsv(path)
    assert np.isnan(db.mw[0])


def test_non_numeric_mass_becomes_nan(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t1.1.1.1\tnot_a_number\tMKAL",
    ])
    db = load_uniprot_tsv(path)
    assert np.isnan(db.mw[0])


def test_short_lines_are_padded(tmp_path):
    # A row with only 3 tab-delimited fields should still parse.
    path = tmp_path / "u.tsv"
    path.write_text(HEADER + "P1\tG1\t1.1.1.1\n")
    db = load_uniprot_tsv(path)
    assert db.ids == ["P1"]
    assert db.sequences == [""]
    assert np.isnan(db.mw[0])


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "u.tsv"
    path.write_text(
        HEADER
        + "P1\tG1\t\t10000\tMKAL\n"
        + "\n"
        + "P2\tG2\t\t20000\tMNTD\n"
    )
    db = load_uniprot_tsv(path)
    assert db.ids == ["P1", "P2"]


# --------------------------------------------------------------------------- #
# Error conditions
# --------------------------------------------------------------------------- #

def test_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_uniprot_tsv(tmp_path / "does_not_exist.tsv")


def test_raises_on_empty_file(tmp_path):
    path = tmp_path / "u.tsv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_uniprot_tsv(path)


def test_raises_on_header_with_too_few_fields(tmp_path):
    path = tmp_path / "u.tsv"
    path.write_text("Entry\tGene\tEC\n")  # only 3 columns
    with pytest.raises(ValueError, match="header"):
        load_uniprot_tsv(path)


def test_raises_on_duplicate_entry(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t\t10000\tMKAL",
        "P2\tG2\t\t20000\tMNTD",
        "P1\tG3\t\t30000\tMSYN",  # duplicate of P1 on a different row
    ])
    with pytest.raises(ValueError, match="Duplicate Entry"):
        load_uniprot_tsv(path)


# --------------------------------------------------------------------------- #
# split_gene_cells option
# --------------------------------------------------------------------------- #

def test_split_gene_cells_default_is_false(tmp_path):
    """Without splitting (the default), multi-gene cells are stored
    verbatim as a single string."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1 G2\t1.1.1.1\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path)
    assert db.genes == ["G1 G2"]
    assert db.ids == ["P1"]


def test_split_gene_cells_expands_multi_gene_rows(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1 G2\t1.1.1.1\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path, split_gene_cells=True)

    assert db.genes == ["G1", "G2"]
    assert db.ids == ["P1", "P1"]
    assert db.eccodes == ["1.1.1.1", "1.1.1.1"]
    np.testing.assert_array_almost_equal(db.mw, [10000.0, 10000.0])
    assert db.sequences == ["MKAL", "MKAL"]


def test_split_gene_cells_with_multiple_whitespace(tmp_path):
    """Tabs and multiple spaces in the Gene Names cell all split."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1  G2   G3\t\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path, split_gene_cells=True)
    assert db.genes == ["G1", "G2", "G3"]


def test_split_gene_cells_empty_cell_stays_empty(tmp_path):
    """An empty Gene Names cell produces a single empty-string row,
    not zero rows. This preserves alignment with the source TSV."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\t\t1.1.1.1\t10000\tMKAL",
    ])
    db = load_uniprot_tsv(path, split_gene_cells=True)
    assert db.genes == [""]
    assert db.ids == ["P1"]


def test_split_gene_cells_does_not_flag_repetition_as_duplicate(tmp_path):
    """Expanded rows share an Entry value, which is expected, not a duplicate."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1 G2\t\t10000\tMKAL",
        "P2\tG3\t\t20000\tMNTD",
    ])
    # Should not raise.
    db = load_uniprot_tsv(path, split_gene_cells=True)
    assert len(db) == 3


def test_split_gene_cells_still_detects_cross_row_duplicates(tmp_path):
    """Duplicate Entry across distinct source rows must still raise."""
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1 G2\t\t10000\tMKAL",
        "P2\tG3\t\t20000\tMNTD",
        "P1\tG4\t\t30000\tMSYN",  # second occurrence of P1
    ])
    with pytest.raises(ValueError, match="Duplicate Entry"):
        load_uniprot_tsv(path, split_gene_cells=True)


# --------------------------------------------------------------------------- #
# find_by_gene / find_by_id
# --------------------------------------------------------------------------- #

def test_find_by_gene_returns_index(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t\t10000\tMKAL",
        "P2\tG2\t\t20000\tMNTD",
    ])
    db = load_uniprot_tsv(path)
    assert db.find_by_gene("G2") == 1
    assert db.find_by_gene("Gxxx") is None


def test_find_by_id_returns_index(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        "P1\tG1\t\t10000\tMKAL",
        "P2\tG2\t\t20000\tMNTD",
    ])
    db = load_uniprot_tsv(path)
    assert db.find_by_id("P2") == 1
    assert db.find_by_id("Pxxx") is None


# --------------------------------------------------------------------------- #
# Quoted cells, read as MATLAB's textscan with %q reads them
# --------------------------------------------------------------------------- #

T = "\t"


def test_quoted_cell_is_given_without_its_quotes(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        'P1\tG1\t"1.1.1.25; 2.5.1.19"\t10000\tMKAL',
    ])
    db = load_uniprot_tsv(path)
    assert db.eccodes == ["1.1.1.25; 2.5.1.19"]
    assert db.genes == ["G1"]
    assert db.sequences == ["MKAL"]


def test_quoted_gene_cell_is_split_without_quotes(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        'P1\t"G1 G2"\t\t10000\tMKAL',
    ])
    db = load_uniprot_tsv(path, split_gene_cells=True)
    assert db.genes == ["G1", "G2"]


def test_quoted_entry_ids_count_as_duplicates(tmp_path):
    path = _write_tsv(tmp_path / "u.tsv", [
        'P1\tG1\t\t10000\tMKAL',
        '"P1"\tG2\t\t10000\tMKAL',
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        load_uniprot_tsv(path)


# The expected rows are what MATLAB's textscan(fid, '%q %q %q %q %q', 'Delimiter', '\t',
# 'HeaderLines', 1) returns for the same text (MATLAB R2024b).
@pytest.mark.parametrize(
    "text, rows",
    [
        (f'P1{T}g1{T}"1.1.1.25; 2.5.1.19"{T}100{T}SEQ\n', [["P1", "g1", "1.1.1.25; 2.5.1.19", "100", "SEQ"]]),
        (f'P2{T}"a ""b"" c"{T}{T}10{T}S\n', [["P2", 'a "b" c', "", "10", "S"]]),
        (f'P3{T}"has{T}tab"{T}"x"{T}5{T}S\n', [["P3", f"has{T}tab", "x", "5", "S"]]),
        (f'P4{T}ab"cd"{T}"x"y{T}7{T}S\n', [["P4", 'ab"cd"', "xy", "7", "S"]]),
        (f'P6{T}" spaced "{T}  lead  {T}4{T}S\n', [["P6", " spaced ", "lead  ", "4", "S"]]),
        (f"P7{T}g\n", [["P7", "g", "", "", ""]]),
        (f'P12{T}"""a"""{T}x{T}1{T}S\n', [["P12", '"a"', "x", "1", "S"]]),
        (f'P13{T}a""b{T}"x\ny"{T}1{T}S\n', [["P13", 'a""b', "x\ny", "1", "S"]]),
        (f'P10{T}"unterminated{T}x{T}1{T}S\n', [["P10", f"unterminated{T}x{T}1{T}S\n", "", "", ""]]),
        (f'A{T}g{T}e{T}5{T}S\nB{T}h\nC{T}i{T}f{T}6{T}T\n',
         [["A", "g", "e", "5", "S"], ["B", "h", "", "", ""], ["C", "i", "f", "6", "T"]]),
        (f"A{T}g{T}e{T}5{T}S{T}extra{T}more\nB{T}h{T}e{T}6{T}T\n",
         [["A", "g", "e", "5", "S"], ["extra", "more", "", "", ""], ["B", "h", "e", "6", "T"]]),
        (f'A{T}"x y"{T}e{T}5{T}S\r\nB{T}h{T}e{T}6{T}T\r\n',
         [["A", "x y", "e", "5", "S"], ["B", "h", "e", "6", "T"]]),
        (f"A{T}g{T}{T}{T}\nB{T}h{T}{T}6{T}\n",
         [["A", "g", "", "", ""], ["B", "h", "", "6", ""]]),
        (f"A{T}g{T}e{T}5{T}S\n   \nB{T}h{T}e{T}6{T}T\n",
         [["A", "g", "e", "5", "S"], ["B", "h", "e", "6", "T"]]),
        (f'A{T}"x"  {T}e{T}5{T}S\n', [["A", "x  ", "e", "5", "S"]]),
        (f'A{T}  "x y"{T}e{T}5{T}S\n', [["A", "x y", "e", "5", "S"]]),
        (f"A{T}g{T}e{T}5{T}S", [["A", "g", "e", "5", "S"]]),
        (f"A{T}'x y'{T}e{T}5{T}S\n", [["A", "'x y'", "e", "5", "S"]]),
    ],
)
def test_read_quoted_tsv_matches_matlab_textscan(text, rows):
    from geckopy.databases.uniprot_loader import read_quoted_tsv

    assert read_quoted_tsv(text, 5) == rows


def test_read_quoted_tsv_skips_header_lines():
    from geckopy.databases.uniprot_loader import read_quoted_tsv

    assert read_quoted_tsv(f"a{T}b\r\nx{T}y\n", 2, header_lines=1) == [["x", "y"]]
