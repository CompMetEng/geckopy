"""Tests for load_brenda_data."""
import logging
import lzma
from pathlib import Path

import pytest

from geckopy.databases import BrendaData, load_brenda_data


# Header lines that load_brenda_data is expected to skip.
_COMMENT_KCAT = "# BRENDA release test generated 2026-05-18 - CC BY 4.0 - kcat in 1/s\n"
_COMMENT_SA = (
    "# BRENDA release test generated 2026-05-18 - CC BY 4.0 - "
    "specific activity in umol/min/mg\n"
)
_COMMENT_MW = (
    "# BRENDA release test generated 2026-05-18 - CC BY 4.0 - "
    "molecular weight in g/mol\n"
)
_HEADER_KCAT = (
    "ec_code\tsubstrate\torganism\tkcat_max\tkcat_median\tn\treferences\n"
)
_HEADER_SA = (
    "ec_code\tsubstrate\torganism\tsa_max\tsa_median\tn\treferences\n"
)
_HEADER_MW = "ec_code\tsubstrate\torganism\tmw\tn\treferences\n"


def _write_brenda_files(
    folder: Path,
    *,
    kcat: str = "",
    sa: str = "",
    mw: str = "",
    with_headers: bool = True,
) -> None:
    """Create the three BRENDA TSVs with optional header lines."""
    kcat_text = (_COMMENT_KCAT + _HEADER_KCAT + kcat) if with_headers else kcat
    sa_text = (_COMMENT_SA + _HEADER_SA + sa) if with_headers else sa
    mw_text = (_COMMENT_MW + _HEADER_MW + mw) if with_headers else mw
    (folder / "kcat.tsv").write_text(kcat_text, encoding="utf-8")
    (folder / "sa.tsv").write_text(sa_text, encoding="utf-8")
    (folder / "mw.tsv").write_text(mw_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_returns_brenda_data_with_split_views(tmp_path):
    _write_brenda_files(tmp_path)
    result = load_brenda_data(tmp_path)
    assert isinstance(result, BrendaData)
    expected_kcat = ["ec_code", "substrate", "organism", "kcat", "n"]
    expected_sa = ["ec_code", "organism", "kcat", "mw", "n"]
    assert list(result.kcat_max.columns) == expected_kcat
    assert list(result.kcat_median.columns) == expected_kcat
    assert list(result.sa_max.columns) == expected_sa
    assert list(result.sa_median.columns) == expected_sa


def test_kcat_for_picks_view_by_aggregation(tmp_path):
    _write_brenda_files(
        tmp_path,
        kcat="1.1.1.1\tm1\torg1\t10\t4\t3\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert result.kcat_for("max").iloc[0]["kcat"] == 10.0
    assert result.kcat_for("median").iloc[0]["kcat"] == 4.0


def test_kcat_for_invalid_aggregation_raises(tmp_path):
    _write_brenda_files(tmp_path)
    result = load_brenda_data(tmp_path)
    with pytest.raises(ValueError, match="aggregation must be"):
        result.kcat_for("mean")
    with pytest.raises(ValueError, match="aggregation must be"):
        result.sa_for("bogus")


def test_all_empty_files_yield_empty_dataframes(tmp_path):
    _write_brenda_files(tmp_path)
    result = load_brenda_data(tmp_path)
    assert result.kcat_max.empty
    assert result.kcat_median.empty
    assert result.sa_max.empty
    assert result.sa_median.empty


# --------------------------------------------------------------------------- #
# Outlier filtering (filter_outlier_kcats)
# --------------------------------------------------------------------------- #


def _kcat_row(ec, organism, kcat):
    return f"{ec}\tm1\t{organism}\t{kcat}\t{kcat}\t1\t*\n"


def test_outlier_row_dropped_by_default(tmp_path):
    # Five real-looking values for the same EC (the reference pool),
    # plus one 8-decades-lower outlier for that same EC.
    kcat = "".join(
        _kcat_row("1.1.1.1", f"org{i}", v)
        for i, v in enumerate([10, 12, 8, 15, 11])
    )
    kcat += _kcat_row("1.1.1.1", "org_outlier", 1e-6)
    _write_brenda_files(tmp_path, kcat=kcat)

    filtered = load_brenda_data(tmp_path)
    assert len(filtered.kcat_max) == 5
    assert 1e-6 not in filtered.kcat_max["kcat"].values

    unfiltered = load_brenda_data(tmp_path, filter_outlier_kcats=False)
    assert len(unfiltered.kcat_max) == 6
    assert 1e-6 in unfiltered.kcat_max["kcat"].values


def test_reference_pool_widens_to_ec_prefix_when_sparse(tmp_path):
    # Only one row for 1.1.1.9 itself (too few for its own reference),
    # but four more under the shared 1.1.1.- prefix -- enough (5) once
    # widened. The lone 1.1.1.9 row is a real outlier against that
    # widened reference and should be dropped.
    kcat = "".join(
        _kcat_row(ec, f"org{i}", v)
        for i, (ec, v) in enumerate([
            ("1.1.1.1", 10), ("1.1.1.2", 12), ("1.1.1.3", 8), ("1.1.1.4", 15),
        ])
    )
    kcat += _kcat_row("1.1.1.9", "org_outlier", 1e-6)
    _write_brenda_files(tmp_path, kcat=kcat)

    filtered = load_brenda_data(tmp_path)
    assert "1.1.1.9" not in filtered.kcat_max["ec_code"].values
    assert len(filtered.kcat_max) == 4


def test_outlier_filter_uses_kcat_and_sa_together_as_one_reference_pool(tmp_path):
    # The reference pool for one EC draws on both the plain kcat table
    # and the SA-derived table; four kcat rows plus one SA row is
    # enough (5) to judge a sixth, wildly-off SA row as an outlier.
    kcat = "".join(
        _kcat_row("2.3.1.297", f"org{i}", v)
        for i, v in enumerate([10, 12, 8, 15])
    )
    sa = (
        "2.3.1.297\t*\torg_good\t2.0\t2.0\t1\t*\n"     # ~ SA*MW = 20, in range
        "2.3.1.297\t*\torg_bad\t1e-8\t1e-8\t1\t*\n"    # ~ SA*MW = 1e-7, outlier
    )
    mw = (
        "2.3.1.297\t*\torg_good\t10000\t1\t*\n"
        "2.3.1.297\t*\torg_bad\t10000\t1\t*\n"
    )
    _write_brenda_files(tmp_path, kcat=kcat, sa=sa, mw=mw)

    filtered = load_brenda_data(tmp_path)
    assert len(filtered.sa_max) == 1
    assert filtered.sa_max.iloc[0]["organism"] == "org_good"


def test_empty_tables_unaffected_by_outlier_filter(tmp_path):
    _write_brenda_files(tmp_path)
    result = load_brenda_data(tmp_path, filter_outlier_kcats=True)
    assert list(result.kcat_max.columns) == [
        "ec_code", "substrate", "organism", "kcat", "n",
    ]
    assert result.kcat_max.empty


# --------------------------------------------------------------------------- #
# Missing files
# --------------------------------------------------------------------------- #

def test_missing_kcat_file_raises(tmp_path):
    (tmp_path / "sa.tsv").write_text("")
    (tmp_path / "mw.tsv").write_text("")
    with pytest.raises(FileNotFoundError, match="kcat.tsv"):
        load_brenda_data(tmp_path)


def test_missing_sa_file_raises(tmp_path):
    (tmp_path / "kcat.tsv").write_text("")
    (tmp_path / "mw.tsv").write_text("")
    with pytest.raises(FileNotFoundError, match="sa.tsv"):
        load_brenda_data(tmp_path)


def test_missing_mw_file_raises(tmp_path):
    (tmp_path / "kcat.tsv").write_text("")
    (tmp_path / "sa.tsv").write_text("")
    with pytest.raises(FileNotFoundError, match="mw.tsv"):
        load_brenda_data(tmp_path)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_brenda_data(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# xz-compressed variant (what the published wheel ships)
# --------------------------------------------------------------------------- #

def test_reads_xz_compressed_files(tmp_path):
    """A folder with only the .tsv.xz variant (as an installed wheel
    ships) loads exactly like plain text."""
    kcat_text = (
        _COMMENT_KCAT + _HEADER_KCAT
        + "1.1.1.1\tm1\torg1\t10\t4\t3\t*\n"
    )
    with lzma.open(tmp_path / "kcat.tsv.xz", "wt", encoding="utf-8") as f:
        f.write(kcat_text)
    with lzma.open(tmp_path / "sa.tsv.xz", "wt", encoding="utf-8") as f:
        f.write(_COMMENT_SA + _HEADER_SA)
    with lzma.open(tmp_path / "mw.tsv.xz", "wt", encoding="utf-8") as f:
        f.write(_COMMENT_MW + _HEADER_MW)

    result = load_brenda_data(tmp_path)
    assert result.kcat_for("max").iloc[0]["kcat"] == 10.0
    assert result.kcat_for("median").iloc[0]["kcat"] == 4.0


def test_prefers_xz_over_plain_text_when_both_present(tmp_path):
    """If a folder somehow has both, the .xz variant wins -- it's what
    an installed wheel ships alongside a stale plain-text leftover."""
    _write_brenda_files(
        tmp_path, kcat="1.1.1.1\tm1\torg1\t99\t99\t1\t*\n",
    )
    with lzma.open(tmp_path / "kcat.tsv.xz", "wt", encoding="utf-8") as f:
        f.write(
            _COMMENT_KCAT + _HEADER_KCAT
            + "1.1.1.1\tm1\torg1\t1\t1\t1\t*\n"
        )

    result = load_brenda_data(tmp_path)
    assert result.kcat_for("max").iloc[0]["kcat"] == 1.0


# --------------------------------------------------------------------------- #
# KCAT file parsing
# --------------------------------------------------------------------------- #

def test_kcat_basic_parse_max_and_median(tmp_path):
    _write_brenda_files(
        tmp_path,
        kcat="1.1.1.1\tm1\torg1\t10\t10\t1\tPMID:1\n",
    )
    result = load_brenda_data(tmp_path)
    assert len(result.kcat_max) == 1
    assert len(result.kcat_median) == 1
    row_max = result.kcat_max.iloc[0]
    row_med = result.kcat_median.iloc[0]
    assert row_max["ec_code"] == "1.1.1.1"
    assert row_max["substrate"] == "m1"
    assert row_max["organism"] == "org1"
    assert row_max["kcat"] == 10.0
    assert int(row_max["n"]) == 1
    assert row_med["kcat"] == 10.0


def test_kcat_no_scaling_applied(tmp_path):
    _write_brenda_files(
        tmp_path,
        kcat="1.1.1.1\tm1\torg1\t42.5\t42.5\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert result.kcat_max.iloc[0]["kcat"] == pytest.approx(42.5)
    assert result.kcat_median.iloc[0]["kcat"] == pytest.approx(42.5)


def test_kcat_views_carry_different_values_from_same_row(tmp_path):
    """A wide row with distinct max and median lands in both views with
    its respective value."""
    _write_brenda_files(
        tmp_path,
        kcat=(
            "1.1.1.1\tm1\torg1\t10\t4\t3\t*\n"
            "2.7.7.7\tm2\torg2\t100\t40\t5\t*\n"
        ),
    )
    result = load_brenda_data(tmp_path)
    assert sorted(result.kcat_max["kcat"]) == [10.0, 100.0]
    assert sorted(result.kcat_median["kcat"]) == [4.0, 40.0]


def test_header_and_comment_lines_skipped(tmp_path):
    """Both the ``#`` release comment and the TSV column-header line are
    skipped."""
    _write_brenda_files(
        tmp_path,
        kcat="1.1.1.1\tethanol\tyeast\t23.5\t23.5\t1\tPMID:1\n",
    )
    result = load_brenda_data(tmp_path)
    assert len(result.kcat_max) == 1
    assert result.kcat_max.iloc[0]["ec_code"] == "1.1.1.1"


# --------------------------------------------------------------------------- #
# Malformed lines
# --------------------------------------------------------------------------- #

def test_malformed_too_few_columns_skipped_with_warning(tmp_path, caplog):
    _write_brenda_files(
        tmp_path,
        kcat=(
            "1.1.1.1\tm1\torg1\t1\t1\t1\t*\n"
            "1.1.1.1\tm2\torg1\n"
            "1.1.1.1\tm3\torg1\t3\t3\t1\t*\n"
        ),
    )
    with caplog.at_level(logging.WARNING):
        result = load_brenda_data(tmp_path)
    assert len(result.kcat_max) == 2
    assert "tab-delimited fields" in caplog.text


def test_malformed_non_numeric_value_skipped_with_warning(tmp_path, caplog):
    _write_brenda_files(
        tmp_path,
        kcat=(
            "1.1.1.1\tm1\torg1\t1\t1\t1\t*\n"
            "1.1.1.1\tm2\torg1\tnot_a_number\t2\t1\t*\n"
            "1.1.1.1\tm3\torg1\t3\t3\t1\t*\n"
        ),
    )
    with caplog.at_level(logging.WARNING):
        result = load_brenda_data(tmp_path)
    assert len(result.kcat_max) == 2
    assert "non-numeric" in caplog.text


# --------------------------------------------------------------------------- #
# SA + MW join
# --------------------------------------------------------------------------- #

def test_sa_mw_join_computes_kcat(tmp_path):
    """SA = 60 [umol/min/mg] = 1 [mmol/s/g]; MW = 1000 [g/mol] = 1 [g/mmol].
    kcat = SA * MW = 1 [1/s]."""
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\torg1\t60\t60\t1\t*\n",
        mw="1.1.1.1\t*\torg1\t1000\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert len(result.sa_max) == 1
    row = result.sa_max.iloc[0]
    assert row["ec_code"] == "1.1.1.1"
    assert row["organism"] == "org1"
    assert row["kcat"] == pytest.approx(1.0)
    assert row["mw"] == pytest.approx(1.0)


def test_sa_views_have_separate_kcat_values_from_max_and_median_columns(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\torg1\t120\t60\t3\t*\n",
        mw="1.1.1.1\t*\torg1\t1000\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert result.sa_max.iloc[0]["kcat"] == pytest.approx(2.0)
    assert result.sa_median.iloc[0]["kcat"] == pytest.approx(1.0)


def test_sa_without_mw_match_dropped(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\torgX\t60\t60\t1\t*\n",
        mw="1.1.1.1\t*\torgY\t1000\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert result.sa_max.empty
    assert result.sa_median.empty


def test_sa_mw_join_case_insensitive_on_organism(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\tORGANISM ONE\t60\t60\t1\t*\n",
        mw="1.1.1.1\t*\torganism one\t1000\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert len(result.sa_max) == 1
    assert len(result.sa_median) == 1


def test_sa_first_mw_match_wins(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\torg1\t60\t60\t1\t*\n",
        mw=(
            "1.1.1.1\t*\torg1\t1000\t1\t*\n"
            "1.1.1.1\t*\torg1\t9000\t1\t*\n"
        ),
    )
    result = load_brenda_data(tmp_path)
    assert len(result.sa_max) == 1
    assert result.sa_max.iloc[0]["mw"] == pytest.approx(1.0)


def test_empty_sa_yields_empty_sa_table_even_with_mw(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="",
        mw="1.1.1.1\t*\torg1\t1000\t1\t*\n",
    )
    result = load_brenda_data(tmp_path)
    assert result.sa_max.empty
    assert result.sa_median.empty


def test_empty_mw_yields_empty_sa_table_even_with_sa(tmp_path):
    _write_brenda_files(
        tmp_path,
        sa="1.1.1.1\t*\torg1\t60\t60\t1\t*\n",
        mw="",
    )
    result = load_brenda_data(tmp_path)
    assert result.sa_max.empty
    assert result.sa_median.empty


# --------------------------------------------------------------------------- #
# Real fixture from examples/ecTestGEM
# --------------------------------------------------------------------------- #

@pytest.fixture
def ec_test_gem_brenda_folder() -> Path:
    return Path(__file__).resolve().parent.parent / "examples" / "ecTestGEM" / "data"


def test_real_fixture_loads_kcat(ec_test_gem_brenda_folder):
    result = load_brenda_data(ec_test_gem_brenda_folder)
    # The fixture has 3 (ec, substrate, organism) triples each with a
    # single source measurement, so max == median per triple.
    assert len(result.kcat_max) == 3
    assert list(result.kcat_max["ec_code"]) == ["1.1.1.1", "1.1.1.1", "1.1.2.2"]
    assert list(result.kcat_max["substrate"]) == ["m1", "m2", "m1"]
    assert list(result.kcat_max["kcat"]) == [1.0, 10.0, 100.0]
    assert list(result.kcat_median["kcat"]) == [1.0, 10.0, 100.0]


def test_real_fixture_sa_table_empty_due_to_mismatched_organisms(
    ec_test_gem_brenda_folder,
):
    """The fixture has SA for `acetobacter pasteurianus` and MW for
    `testus testus` - different organisms, so no SA row joins."""
    result = load_brenda_data(ec_test_gem_brenda_folder)
    assert result.sa_max.empty
    assert result.sa_median.empty
