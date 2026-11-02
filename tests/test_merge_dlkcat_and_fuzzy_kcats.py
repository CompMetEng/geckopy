"""Tests for merge_kcats and the deprecated merge_dlkcat_and_fuzzy_kcats."""
import pandas as pd
import pytest

from geckopy.gather_kcats import (
    merge_dlkcat_and_fuzzy_kcats,
    merge_kcats,
    normalize_source,
)

# The wrapper is intentionally deprecated; silence its warning so the
# legacy-behaviour tests below stay quiet.
pytestmark = pytest.mark.filterwarnings(
    "ignore:merge_dlkcat_and_fuzzy_kcats is deprecated:DeprecationWarning"
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

_CANONICAL_COLUMNS = [
    "rxn_id", "source", "eccode", "substrates", "genes",
    "kcat", "wildcard_level", "origin",
]


def _fuzzy_row(
    rxn_id: str,
    kcat: float,
    *,
    wildcard_level: object,
    origin: object,
    eccode: str = "1.1.1.1",
    substrates: list[str] | None = None,
) -> dict:
    return {
        "rxn_id": rxn_id,
        "source": "brenda",
        "eccode": eccode,
        "substrates": substrates if substrates is not None else ["alpha"],
        "genes": [],
        "kcat": kcat,
        "wildcard_level": wildcard_level,
        "origin": origin,
    }


def _dlkcat_row(rxn_id: str, kcat: float, gene: str = "g1") -> dict:
    return {
        "rxn_id": rxn_id,
        "source": "DLKcat",
        "eccode": "",
        "substrates": ["alpha"],
        "genes": [gene],
        "kcat": kcat,
        "wildcard_level": pd.NA,
        "origin": pd.NA,
    }


def _fuzzy_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS) if rows else pd.DataFrame(
        columns=_CANONICAL_COLUMNS,
    )
    if "wildcard_level" in df:
        df["wildcard_level"] = df["wildcard_level"].astype("Int64")
    if "origin" in df:
        df["origin"] = df["origin"].astype("Int64")
    return df


def _dlkcat_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS) if rows else pd.DataFrame(
        columns=_CANONICAL_COLUMNS,
    )
    if "wildcard_level" in df:
        df["wildcard_level"] = df["wildcard_level"].astype("Int64")
    if "origin" in df:
        df["origin"] = df["origin"].astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("top", [0, 7, -1])
def test_top_origin_out_of_range_raises(top):
    with pytest.raises(ValueError, match="top_origin_limit"):
        merge_dlkcat_and_fuzzy_kcats(
            _dlkcat_df([]), _fuzzy_df([]), top_origin_limit=top,
        )


@pytest.mark.parametrize("bot", [0, 7])
def test_bottom_origin_out_of_range_raises(bot):
    with pytest.raises(ValueError, match="bottom_origin_limit"):
        merge_dlkcat_and_fuzzy_kcats(
            _dlkcat_df([]), _fuzzy_df([]), bottom_origin_limit=bot,
        )


@pytest.mark.parametrize("wc", [-1, 4])
def test_wildcard_limit_out_of_range_raises(wc):
    with pytest.raises(ValueError, match="wildcard_limit"):
        merge_dlkcat_and_fuzzy_kcats(
            _dlkcat_df([]), _fuzzy_df([]), wildcard_limit=wc,
        )


def test_missing_required_column_raises():
    bad_df = pd.DataFrame({"rxn_id": ["r1"]})  # missing most columns
    with pytest.raises(ValueError, match="missing required"):
        merge_dlkcat_and_fuzzy_kcats(bad_df, _fuzzy_df([]))


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_both_empty_returns_empty():
    out = merge_dlkcat_and_fuzzy_kcats(_dlkcat_df([]), _fuzzy_df([]))
    assert out.empty
    assert list(out.columns) == _CANONICAL_COLUMNS


def test_only_dlkcat_input():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 5.0), _dlkcat_row("r2", 7.0)]),
        _fuzzy_df([]),
    )
    assert sorted(out["rxn_id"]) == ["r1", "r2"]
    assert all(out["source"] == "DLKcat")


def test_only_fuzzy_input():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)]),
    )
    assert list(out["rxn_id"]) == ["r1"]
    assert list(out["source"]) == ["brenda"]


# --------------------------------------------------------------------------- #
# Prio 1: fuzzy wc=0 + origin<=top wins over dlkcat
# --------------------------------------------------------------------------- #

def test_fuzzy_prio1_wins_when_reaction_also_in_dlkcat():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)]),
        top_origin_limit=3,
    )
    # Fuzzy prio1 wins: only one row for r1, from fuzzy.
    assert len(out) == 1
    row = out.iloc[0]
    assert row["source"] == "brenda"
    assert row["kcat"] == 5.0


def test_fuzzy_prio1_uses_default_top_origin_limit_4():
    """Default top_origin_limit=4: an origin=6 (specific-activity x MW
    derived) match does NOT qualify for prio1, so DLKcat wins instead.
    Origins 5-6 are demoted by default because BRENDA's SA/MW join can
    produce a wildly wrong outlier that would otherwise silently
    outrank a good prediction -- see brenda_loader's
    filter_outlier_kcats, which this default is meant to pair with."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=6)]),
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


def test_fuzzy_prio1_origin_6_qualifies_when_limit_explicitly_widened():
    """The pre-4-default behaviour is still reachable by passing
    top_origin_limit=6 explicitly."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=6)]),
        top_origin_limit=6,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "brenda"


def test_fuzzy_with_wildcard_does_not_qualify_for_prio1():
    """Wildcard fuzzy is NOT prio1; DLKcat wins."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=1, origin=1)]),
        top_origin_limit=3,
        wildcard_limit=0,  # prevent prio3 by setting wildcard_limit too low
    )
    # No prio1 (wc != 0), no prio3 (wc > wildcard_limit) -> only DLKcat.
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


# --------------------------------------------------------------------------- #
# Prio 2: DLKcat fills in reactions not in fuzzy prio1
# --------------------------------------------------------------------------- #

def test_dlkcat_prio2_when_fuzzy_origin_too_high():
    """Fuzzy origin > top_origin_limit -> not prio1.
    With bottom_origin_limit < fuzzy origin -> not prio3 either.
    So DLKcat wins (prio2)."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=6)]),
        top_origin_limit=3,
        bottom_origin_limit=3,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


def test_dlkcat_prio2_when_fuzzy_missing():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r2", 5.0, wildcard_level=0, origin=1)]),
    )
    # r1 only in DLKcat, r2 only in fuzzy prio1 -> both kept.
    assert sorted(out["rxn_id"]) == ["r1", "r2"]


# --------------------------------------------------------------------------- #
# Prio 3: fuzzy fallback when DLKcat doesn't cover it
# --------------------------------------------------------------------------- #

def test_fuzzy_prio3_via_origin_branch():
    """top < origin <= bottom, wc = 0 -> prio3.
    DLKcat does NOT cover r1, so fuzzy prio3 wins."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=5)]),
        top_origin_limit=3,
        bottom_origin_limit=5,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "brenda"
    assert out.iloc[0]["kcat"] == 5.0


def test_fuzzy_prio3_via_wildcard_branch():
    """0 < wc <= wildcard_limit AND origin <= bottom -> prio3."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=2, origin=3)]),
        top_origin_limit=3,
        bottom_origin_limit=6,
        wildcard_limit=3,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "brenda"


def test_fuzzy_prio3_excluded_when_wildcard_too_high():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=3, origin=1)]),
        wildcard_limit=2,
    )
    assert out.empty


def test_fuzzy_prio3_excluded_when_origin_above_bottom():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=1, origin=6)]),
        bottom_origin_limit=3,
    )
    assert out.empty


def test_dlkcat_prio2_wins_over_fuzzy_prio3():
    """When the same reaction has both a prio3 fuzzy row AND a DLKcat row,
    DLKcat (prio2) wins."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=1, origin=3)]),
        top_origin_limit=3,
        bottom_origin_limit=6,
        wildcard_limit=3,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #

def test_fuzzy_row_with_na_origin_excluded_from_all_priorities():
    """A fuzzy row that didn't match anything (NaN origin / NaN wildcard)
    must be excluded from prio1 and prio3 alike."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 0.0, wildcard_level=pd.NA, origin=pd.NA)]),
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


# --------------------------------------------------------------------------- #
# Ordering and schema
# --------------------------------------------------------------------------- #

def test_output_schema_matches_inputs():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r2", 5.0, wildcard_level=0, origin=1)]),
    )
    assert list(out.columns) == _CANONICAL_COLUMNS


def test_fuzzy_rows_come_before_dlkcat_rows():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _fuzzy_df([_fuzzy_row("r2", 5.0, wildcard_level=0, origin=1)]),
    )
    sources = list(out["source"])
    # First all "brenda" entries, then all "DLKcat" entries.
    first_dlkcat = next((i for i, s in enumerate(sources) if s == "DLKcat"), len(sources))
    assert all(s == "brenda" for s in sources[:first_dlkcat])
    assert all(s == "DLKcat" for s in sources[first_dlkcat:])


# --------------------------------------------------------------------------- #
# Multiple rows per reaction
# --------------------------------------------------------------------------- #

def test_multiple_dlkcat_rows_per_reaction_all_kept_when_prio2():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([
            _dlkcat_row("r1", 100.0),
            _dlkcat_row("r1", 200.0, gene="g2"),
        ]),
        _fuzzy_df([]),
    )
    assert len(out) == 2
    assert all(out["rxn_id"] == "r1")


def test_multiple_fuzzy_rows_per_reaction_all_kept_when_prio1():
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([]),
        _fuzzy_df([
            _fuzzy_row("r1", 5.0, wildcard_level=0, origin=1, eccode="1.1.1.1"),
            _fuzzy_row("r1", 8.0, wildcard_level=0, origin=1, eccode="2.7.7.7"),
        ]),
    )
    assert len(out) == 2
    assert all(out["rxn_id"] == "r1")
    assert sorted(out["kcat"]) == [5.0, 8.0]


def test_fuzzy_prio1_row_blocks_dlkcat_rows_for_same_reaction():
    """One fuzzy prio1 row for r1 makes ALL DLKcat rows for r1
    drop out, even though there are multiple."""
    out = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([
            _dlkcat_row("r1", 100.0),
            _dlkcat_row("r1", 200.0, gene="g2"),
        ]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)]),
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "brenda"


# --------------------------------------------------------------------------- #
# Realistic end-to-end with apply_kcat_list
# --------------------------------------------------------------------------- #

def test_output_can_feed_apply_kcat_list(tmp_path):
    """End-to-end: a merged DataFrame should pass through
    apply_kcat_list without column adjustments."""
    import numpy as np
    from scipy import sparse

    from geckopy.ec_model import EcModel
    from geckopy.ec_model.ec_data import EcData
    from geckopy.gather_kcats import apply_kcat_list

    merged = merge_dlkcat_and_fuzzy_kcats(
        _dlkcat_df([_dlkcat_row("r2", 100.0)]),
        _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)]),
    )

    model = EcModel("test")
    model.ec = EcData(
        rxns=["r1", "r2"],
        kcat=np.array([np.nan, np.nan]),
        source=["", ""],
        notes=["", ""],
        eccodes=["", ""],
        rxn_enz_mat=sparse.csr_matrix((2, 0), dtype=float),
    )
    updated = apply_kcat_list(model, merged)
    assert sorted(updated) == ["r1", "r2"]
    # Fuzzy match -> wildcard/origin detail; DLKcat -> bare lowercase token.
    assert model.ec.source[0] == "brenda (wc=0, origin=1)"
    assert model.ec.source[1] == "dlkcat"


# --------------------------------------------------------------------------- #
# merge_kcats: generalized merge over arbitrary / mixed sources
# --------------------------------------------------------------------------- #

def _okp_row(
    rxn_id: str,
    kcat: float,
    source: str,
    *,
    eccode: str = "",
    gene: str = "g1",
) -> dict:
    """A row as produced by parse_okp_output: per-row provenance source
    tagged with the OKP- pipeline-stage prefix, NA wildcard/origin."""
    return {
        "rxn_id": rxn_id,
        "source": f"OKP-{source}",
        "eccode": eccode,
        "substrates": ["alpha"],
        "genes": [gene],
        "kcat": kcat,
        "wildcard_level": pd.NA,
        "origin": pd.NA,
    }


def _okp_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS) if rows else pd.DataFrame(
        columns=_CANONICAL_COLUMNS,
    )
    df["wildcard_level"] = df["wildcard_level"].astype("Int64")
    df["origin"] = df["origin"].astype("Int64")
    return df


def test_normalize_source_folds_to_snake_case():
    assert normalize_source("Sabio-RK") == "sabio_rk"
    assert normalize_source("DLKcat") == "dlkcat"
    assert normalize_source("CataPro") == "catapro"
    assert normalize_source("BRENDA") == "brenda"
    assert normalize_source("brenda") == "brenda"


def test_merge_kcats_requires_at_least_one_list():
    with pytest.raises(ValueError, match="at least one"):
        merge_kcats(source_priority=["dlkcat"])


def test_merge_kcats_empty_source_priority_raises():
    with pytest.raises(ValueError, match="source_priority"):
        merge_kcats(_okp_df([]), source_priority=[])


def test_merge_kcats_missing_column_raises():
    bad = pd.DataFrame({"rxn_id": ["r1"]})
    with pytest.raises(ValueError, match="missing required"):
        merge_kcats(bad, source_priority=["dlkcat"])


def test_merge_kcats_okp_experimental_db_is_database_exact():
    """An OKP BRENDA value (positive kcat, NA metadata) is an exact DB
    hit (database_exact), beating a lower-priority CataPro prediction."""
    okp = _okp_df([
        _okp_row("r1", 7.0, "BRENDA"),
        _okp_row("r1", 99.0, "CataPro"),
    ])
    out = merge_kcats(okp, source_priority=["database_exact", "catapro"])
    assert len(out) == 1
    assert out.iloc[0]["source"] == "OKP-BRENDA"
    assert out.iloc[0]["kcat"] == 7.0


def test_merge_kcats_sabio_rk_is_database_exact():
    okp = _okp_df([
        _okp_row("r1", 4.0, "Sabio-RK"),
        _okp_row("r1", 99.0, "CataPro"),
    ])
    out = merge_kcats(okp, source_priority=["database_exact", "catapro"])
    assert len(out) == 1
    # Original (un-normalised, OKP--tagged) source label is preserved on
    # output.
    assert out.iloc[0]["source"] == "OKP-Sabio-RK"


def test_merge_kcats_exact_db_preferred_over_fuzzy_top():
    """An exact OKP BRENDA hit outranks a fuzzy BRENDA top match for the
    same reaction when database_exact precedes database_top."""
    fuzzy = _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)])
    okp = _okp_df([_okp_row("r1", 8.0, "BRENDA")])
    out = merge_kcats(
        fuzzy, okp,
        source_priority=["database_exact", "database_top", "database_bottom"],
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "OKP-BRENDA"
    assert out.iloc[0]["kcat"] == 8.0


def test_merge_kcats_okp_brenda_beats_dlkcat():
    """An OKP BRENDA row carries a positive kcat, so it is a real
    database_exact value that outranks a lower-priority DLKcat prediction."""
    out = merge_kcats(
        _okp_df([_okp_row("r1", 7.0, "BRENDA")]),
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        source_priority=["database_exact", "dlkcat"],
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "OKP-BRENDA"


def test_merge_kcats_catapro_used_when_no_database_value():
    okp = _okp_df([
        _okp_row("r1", 7.0, "BRENDA"),
        _okp_row("r2", 50.0, "CataPro"),
    ])
    out = merge_kcats(okp, source_priority=["database_exact", "catapro"])
    assert sorted(out["rxn_id"]) == ["r1", "r2"]
    by_rxn = {r["rxn_id"]: r["source"] for _, r in out.iterrows()}
    assert by_rxn == {"r1": "OKP-BRENDA", "r2": "OKP-CataPro"}


def test_merge_kcats_fuzzy_and_okp_mixed():
    """Fuzzy BRENDA (metadata -> database_top), an OKP Sabio-RK exact hit
    (database_exact) and a CataPro prediction merge together, each
    winning its own reaction."""
    fuzzy = _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=0, origin=1)])
    okp = _okp_df([
        _okp_row("r2", 8.0, "Sabio-RK"),
        _okp_row("r3", 60.0, "CataPro"),
    ])
    out = merge_kcats(
        fuzzy, okp,
        source_priority=["database_exact", "database_top", "catapro"],
    )
    by_rxn = {r["rxn_id"]: r["source"] for _, r in out.iterrows()}
    assert by_rxn == {"r1": "brenda", "r2": "OKP-Sabio-RK", "r3": "OKP-CataPro"}


def test_merge_kcats_keeps_all_rows_of_winning_tier():
    okp = _okp_df([
        _okp_row("r1", 7.0, "BRENDA", gene="g1"),
        _okp_row("r1", 9.0, "BRENDA", gene="g2"),
        _okp_row("r1", 99.0, "CataPro"),
    ])
    out = merge_kcats(okp, source_priority=["database_exact", "catapro"])
    assert len(out) == 2
    assert sorted(out["kcat"]) == [7.0, 9.0]
    assert all(out["source"] == "OKP-BRENDA")


def test_merge_kcats_database_bottom_fallback():
    """A weak fuzzy match (wildcard) is database_bottom and only wins when
    no higher tier covers the reaction."""
    fuzzy = _fuzzy_df([_fuzzy_row("r1", 5.0, wildcard_level=2, origin=3)])
    out = merge_kcats(
        fuzzy,
        source_priority=["database_top", "dlkcat", "database_bottom"],
        top_origin_limit=3,
        bottom_origin_limit=6,
        wildcard_limit=3,
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "brenda"


def test_merge_kcats_unknown_source_dropped_with_warning(caplog):
    """A source not listed in source_priority is dropped and warned about."""
    okp = _okp_df([
        _okp_row("r1", 7.0, "BRENDA"),
        _okp_row("r2", 50.0, "CataPro"),
    ])
    import logging

    with caplog.at_level(logging.WARNING):
        out = merge_kcats(okp, source_priority=["database_exact"])
    assert list(out["rxn_id"]) == ["r1"]
    assert "catapro" in caplog.text


def test_merge_kcats_preserves_input_order():
    okp = _okp_df([
        _okp_row("r2", 50.0, "CataPro"),
        _okp_row("r1", 7.0, "BRENDA"),
    ])
    out = merge_kcats(okp, source_priority=["database_exact", "catapro"])
    # Row order follows the input list (matching MATLAB mergeKcats), not
    # the tier priority: CataPro was first in, so it stays first.
    assert list(out["source"]) == ["OKP-CataPro", "OKP-BRENDA"]


def test_merge_kcats_drops_nonpositive_kcat():
    """An unmatched fuzzy row (kcat 0, NA metadata) is dropped, so a
    DLKcat row for the same reaction wins."""
    out = merge_kcats(
        _fuzzy_df([_fuzzy_row("r1", 0.0, wildcard_level=pd.NA, origin=pd.NA)]),
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        source_priority=["database_top", "dlkcat", "database_bottom"],
    )
    assert len(out) == 1
    assert out.iloc[0]["source"] == "DLKcat"


def test_merge_kcats_okp_prefix_stripped_for_source_priority_match():
    """A non-database OKP source (e.g. CataPro) matches its bare name in
    source_priority despite the OKP- tag, exactly like a database source
    strips the tag for tiering."""
    out = merge_kcats(
        _okp_df([_okp_row("r1", 7.0, "CataPro")]),
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        source_priority=["catapro", "dlkcat"],
    )
    assert len(out) == 1
    # The tag is stripped only for matching; the output keeps it.
    assert out.iloc[0]["source"] == "OKP-CataPro"
    assert out.iloc[0]["kcat"] == 7.0


def test_merge_kcats_okp_and_bare_source_rank_identically():
    """An OKP-tagged row and a bare row for the same underlying source
    are indistinguishable for ranking purposes -- only the tag differs
    in the output."""
    bare_out = merge_kcats(
        _dlkcat_df([_dlkcat_row("r1", 100.0)]),
        _okp_df([_okp_row("r2", 100.0, "DLKcat")]),
        source_priority=["dlkcat"],
    )
    assert sorted(bare_out["rxn_id"]) == ["r1", "r2"]
    assert len(bare_out) == 2
