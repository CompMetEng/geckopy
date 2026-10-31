"""The kcat review functions are public API of geckopy and geckopy.gather_kcats."""
import geckopy
from geckopy import gather_kcats
from geckopy.databases import load_brenda_data

_NAMES = ("EcStats", "ec_stats_from_brenda", "review_assignment")


def test_review_functions_are_exported_from_both_packages():
    for name in _NAMES:
        assert name in geckopy.__all__
        assert name in gather_kcats.__all__
        assert getattr(geckopy, name) is getattr(gather_kcats, name)


def test_ec_stats_keep_the_case_of_the_ec_number(tmp_path):
    (tmp_path / "kcat.tsv").write_text(
        "# BRENDA release test\n"
        "ec_code\tsubstrate\torganism\tkcat_max\tkcat_median\tn\treferences\n"
        "3.5.1.n3\tm1\torg\t40\t4\t2\t*\n"
        "1.1.1.1\tm1\torg\t10\t2\t2\t*\n"
        "1.1.1.1\tm2\torg\t6\t3\t2\t*\n",
        encoding="utf-8",
    )
    (tmp_path / "sa.tsv").write_text(
        "# BRENDA release test\n"
        "ec_code\tsubstrate\torganism\tsa_max\tsa_median\tn\treferences\n",
        encoding="utf-8",
    )
    (tmp_path / "mw.tsv").write_text(
        "# BRENDA release test\nec_code\tsubstrate\torganism\tmw\tn\treferences\n",
        encoding="utf-8",
    )
    stats = geckopy.ec_stats_from_brenda(load_brenda_data(tmp_path, filter_outlier_kcats=False))
    assert sorted(stats) == ["1.1.1.1", "3.5.1.n3"]
    assert stats["1.1.1.1"].kcat_max == 10.0
    assert stats["1.1.1.1"].kcat_median == 2.5
    assert stats["3.5.1.n3"].values == frozenset({4.0, 40.0})
    findings = geckopy.review_assignment(
        ["R1"], [40.0], ["brenda"], ["3.5.1.n3"], ec_stats=stats,
    )
    assert [f.checks for f in findings] == [("ec-maximum",)]
