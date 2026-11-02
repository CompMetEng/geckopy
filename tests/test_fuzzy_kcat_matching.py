"""Tests for fuzzy_kcat_matching."""
import logging
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.databases import BrendaData, PhylDist
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import fuzzy_kcat_matching
from geckopy.gather_kcats.fuzzy_kcat_matching import (
    apply_force_wildcards,
    build_ec_indices,
    escalate_wildcard,
    find_ec_rows,
    resolve_organism_index,
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _adapter_with_org(tmp_path: Path, org_name: str) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\norg_name = "{org_name}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _ec_model(
    adapter: ModelAdapter,
    rxn_specs: list[tuple[str, str, list[tuple[str, float]]]],
    *,
    gecko_light: bool = False,
    ec_rxn_prefix: str = "",
) -> EcModel:
    """Build EcModel where each rxn_spec is
    (rxn_id, eccode, [(met_name, coeff), ...]).

    Negative coeffs are reactants (substrates) for the matching code.
    """
    model = EcModel("test", adapter=adapter, gecko_light=gecko_light)
    mets: dict[str, cobra.Metabolite] = {}
    for _, _, met_list in rxn_specs:
        for name, _ in met_list:
            if name not in mets:
                m = cobra.Metabolite(name, compartment="c")
                m.name = name
                mets[name] = m
    model.add_metabolites(list(mets.values()))

    for rxn_id, _, met_list in rxn_specs:
        rxn = cobra.Reaction(rxn_id)
        rxn.lower_bound = 0.0
        rxn.upper_bound = 1000.0
        rxn.add_metabolites({mets[name]: c for name, c in met_list})
        model.add_reactions([rxn])

    n = len(rxn_specs)
    model.ec = EcData(
        gecko_light=gecko_light,
        rxns=[ec_rxn_prefix + r for r, _, _ in rxn_specs],
        kcat=np.full(n, np.nan, dtype=float),
        source=[""] * n,
        notes=[""] * n,
        eccodes=[ec for _, ec, _ in rxn_specs],
        rxn_enz_mat=sparse.csr_matrix((n, 0), dtype=float),
    )
    return model


def _brenda(
    kcat_rows: list[tuple[str, str, str, float]] | None = None,
    sa_rows: list[tuple[str, str, float, float]] | None = None,
) -> BrendaData:
    """kcat_rows: (ec, substrate, organism, kcat 1/s).
    sa_rows: (ec, organism, kcat 1/s, mw g/mmol)."""
    kcat_rows = kcat_rows or []
    sa_rows = sa_rows or []
    kcat_df = pd.DataFrame(
        kcat_rows, columns=["ec_code", "substrate", "organism", "kcat"]
    )
    sa_df = pd.DataFrame(
        sa_rows, columns=["ec_code", "organism", "kcat", "mw"]
    )
    # The same rows fill both the max and median views; fuzzy_kcat_matching
    # picks the view based on the `aggregate` arg / adapter setting, so
    # tests that don't toggle aggregate behave identically across both.
    return BrendaData(
        kcat_max=kcat_df, kcat_median=kcat_df,
        sa_max=sa_df, sa_median=sa_df,
    )


def _phyl_dist(
    names: list[str], dist: np.ndarray | None = None,
) -> PhylDist:
    n = len(names)
    if dist is None:
        dist = np.zeros((n, n), dtype=float)
    pd_obj = PhylDist(names=list(names), dist_matrix=np.asarray(dist, dtype=float))
    for i, name in enumerate(names):
        lower = name.lower()
        pd_obj.name_to_index.setdefault(lower, i)
        parts = lower.split(None, 1)
        if parts:
            pd_obj.genus_to_indices.setdefault(parts[0], []).append(i)
    return pd_obj


# --------------------------------------------------------------------------- #
# Pure helper tests
# --------------------------------------------------------------------------- #

def test_escalate_wildcard_rightmost_first():
    assert escalate_wildcard("1.2.3.4") == "1.2.3.-"
    assert escalate_wildcard("1.2.3.-") == "1.2.-.-"
    assert escalate_wildcard("1.2.-.-") == "1.-.-.-"
    assert escalate_wildcard("1.-.-.-") == "-.-.-.-"
    assert escalate_wildcard("-.-.-.-") is None


def test_escalate_wildcard_invalid_arity_returns_none():
    assert escalate_wildcard("1.2.3") is None
    assert escalate_wildcard("1.2.3.4.5") is None


def test_apply_force_wildcards_zero_is_noop():
    assert apply_force_wildcards("1.2.3.4", 0) == "1.2.3.4"


def test_apply_force_wildcards_n_steps():
    assert apply_force_wildcards("1.2.3.4", 1) == "1.2.3.-"
    assert apply_force_wildcards("1.2.3.4", 2) == "1.2.-.-"
    assert apply_force_wildcards("1.2.3.4", 4) == "-.-.-.-"
    assert apply_force_wildcards("1.2.3.4", 5) == "-.-.-.-"


def test_build_ec_indices_groups_by_lowercase():
    df = pd.DataFrame({
        "ec_code": ["1.1.1.1", "1.1.1.1", "2.7.7.7"],
        "substrate": ["a", "b", "c"],
        "organism": ["x", "y", "z"],
        "kcat": [1.0, 2.0, 3.0],
    })
    idx = build_ec_indices(df)
    assert sorted(idx.keys()) == ["1.1.1.1", "2.7.7.7"]
    assert sorted(idx["1.1.1.1"].tolist()) == [0, 1]


def test_find_ec_rows_exact_match():
    df = pd.DataFrame({
        "ec_code": ["1.1.1.1", "2.7.7.7"],
        "substrate": ["a", "b"],
        "organism": ["x", "y"],
        "kcat": [1.0, 2.0],
    })
    idx = build_ec_indices(df)
    assert find_ec_rows("1.1.1.1", idx).tolist() == [0]
    assert find_ec_rows("2.7.7.7", idx).tolist() == [1]
    assert find_ec_rows("9.9.9.9", idx).tolist() == []


def test_find_ec_rows_wildcard_prefix_match():
    df = pd.DataFrame({
        "ec_code": ["1.1.1.1", "1.1.1.2", "1.1.2.1", "2.7.7.7"],
        "substrate": ["a"] * 4,
        "organism": ["x"] * 4,
        "kcat": [1.0] * 4,
    })
    idx = build_ec_indices(df)
    # Last-level wildcard: matches both 1.1.1.1 and 1.1.1.2 (prefix "1.1.1.")
    assert sorted(find_ec_rows("1.1.1.-", idx).tolist()) == [0, 1]
    # Two-level wildcard: 1.1.* prefix matches 1.1.1.1, 1.1.1.2, 1.1.2.1
    assert sorted(find_ec_rows("1.1.-.-", idx).tolist()) == [0, 1, 2]


def test_find_ec_rows_full_wildcard_matches_all():
    df = pd.DataFrame({
        "ec_code": ["1.1.1.1", "2.7.7.7", "3.4.21.1"],
        "substrate": ["a", "b", "c"],
        "organism": ["x", "y", "z"],
        "kcat": [1.0, 2.0, 3.0],
    })
    idx = build_ec_indices(df)
    assert sorted(find_ec_rows("-.-.-.-", idx).tolist()) == [0, 1, 2]


def test_resolve_organism_index_direct_hit():
    pd_obj = _phyl_dist(["Saccharomyces cerevisiae", "Escherichia coli"])
    assert resolve_organism_index("Saccharomyces cerevisiae", pd_obj) == 0


def test_resolve_organism_index_genus_fallback():
    pd_obj = _phyl_dist(["Saccharomyces cerevisiae", "Escherichia coli"])
    # Direct miss but genus matches.
    assert resolve_organism_index("Saccharomyces foo", pd_obj) == 0


def test_resolve_organism_index_no_match():
    pd_obj = _phyl_dist(["Escherichia coli"])
    assert resolve_organism_index("totally unknown", pd_obj) is None


def test_resolve_organism_index_empty_name():
    pd_obj = _phyl_dist(["Escherichia coli"])
    assert resolve_organism_index("", pd_obj) is None


# --------------------------------------------------------------------------- #
# Trivial public-function cases
# --------------------------------------------------------------------------- #

def test_empty_model_returns_empty_df(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [])
    df = fuzzy_kcat_matching(model, _brenda(), _phyl_dist(["yeast"]))
    assert df.empty
    assert list(df.columns) == [
        "rxn_id", "source", "eccode", "substrates", "genes", "kcat",
        "wildcard_level", "origin",
    ]


def test_no_adapter_raises(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        fuzzy_kcat_matching(model, _brenda(), _phyl_dist(["yeast"]))


def test_negative_force_wildcard_raises(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [])
    with pytest.raises(ValueError, match="force_wildcard_level"):
        fuzzy_kcat_matching(
            model, _brenda(), _phyl_dist(["yeast"]),
            force_wildcard_level=-1,
        )


def test_unknown_ec_rxn_id_raises(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    with pytest.raises(ValueError, match="not present in model.ec.rxns"):
        fuzzy_kcat_matching(
            model, _brenda(), _phyl_dist(["yeast"]),
            ec_rxns=["nonexistent"],
        )


def test_empty_eccode_yields_zero_kcat_row(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model, _brenda(kcat_rows=[("1.1.1.1", "alpha", "yeast", 5.0)]),
        _phyl_dist(["yeast"]),
    )
    assert len(df) == 1
    assert df.iloc[0]["kcat"] == 0.0
    assert pd.isna(df.iloc[0]["origin"])
    assert pd.isna(df.iloc[0]["wildcard_level"])


# --------------------------------------------------------------------------- #
# Each of the 6 origins triggered in isolation
# --------------------------------------------------------------------------- #

def test_origin_1_org_subs_kcat(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "yeast", 5.0),
            ("1.1.1.1", "alpha", "ecoli", 99.0),
        ]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(5.0)
    assert row["origin"] == 1
    assert row["wildcard_level"] == 0


def test_origin_2_any_org_subs_kcat(tmp_path):
    """No yeast match for substrate; ecoli has a substrate match."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "ecoli", 7.0),
        ]),
        _phyl_dist(["yeast", "ecoli"], dist=np.array([[0, 5], [5, 0]])),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(7.0)
    assert row["origin"] == 2


def test_origin_3_org_no_subs_kcat(tmp_path):
    """Yeast match exists but for a different substrate."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "different_substrate", "yeast", 11.0),
        ]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(11.0)
    assert row["origin"] == 3


def test_origin_4_any_org_no_subs_kcat(tmp_path):
    """No yeast match at all; ecoli has a non-substrate-matching kcat."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "different_substrate", "ecoli", 13.0),
        ]),
        _phyl_dist(["yeast", "ecoli"], dist=np.array([[0, 5], [5, 0]])),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(13.0)
    assert row["origin"] == 4


def test_origin_5_org_sa(tmp_path):
    """Yeast not in KCAT but in SA."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(sa_rows=[("1.1.1.1", "yeast", 17.0, 50.0)]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(17.0)
    assert row["origin"] == 5


def test_origin_6_any_org_sa(tmp_path):
    """No yeast at all; ecoli SA."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(sa_rows=[("1.1.1.1", "ecoli", 19.0, 50.0)]),
        _phyl_dist(["yeast", "ecoli"], dist=np.array([[0, 5], [5, 0]])),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(19.0)
    assert row["origin"] == 6


# --------------------------------------------------------------------------- #
# MATLAB search-order quirk: org-SA is tried before any-no-subs-kcat
# --------------------------------------------------------------------------- #

def test_org_sa_wins_over_any_org_kcat_when_both_present(tmp_path):
    """When both an org-SA row (origin 5) and an any-org-no-subs-kcat
    row (origin 4) are available, the org-SA match wins."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(
            kcat_rows=[("1.1.1.1", "different_substrate", "ecoli", 100.0)],
            sa_rows=[("1.1.1.1", "yeast", 23.0, 50.0)],
        ),
        _phyl_dist(["yeast", "ecoli"], dist=np.array([[0, 5], [5, 0]])),
    )
    row = df.iloc[0]
    assert row["origin"] == 5  # org-SA wins
    assert row["kcat"] == pytest.approx(23.0)


# --------------------------------------------------------------------------- #
# Wildcard escalation
# --------------------------------------------------------------------------- #

def test_wildcard_escalates_when_exact_ec_misses(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.99", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            # Different last-level EC; only matches when wildcard is applied.
            ("1.1.1.1", "anything", "yeast", 31.0),
        ]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    # Substrate doesn't match either ("anything" vs "alpha"), and a wildcard
    # was needed, so origin is 3 (org-no-subs-kcat) at wildcard level 1.
    assert row["wildcard_level"] == 1
    assert row["origin"] == 3
    assert row["kcat"] == pytest.approx(31.0)


def test_wildcard_disables_substrate_match(tmp_path):
    """Even if substrate would match exactly, when wildcard escalation
    kicks in we skip origin-1/2 (substrate-matching levels)."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.99", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "yeast", 37.0),
        ]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    # Origin should be 3 (org-no-subs-kcat), NOT 1 (which would require
    # substrate matching).
    assert row["origin"] == 3
    assert row["wildcard_level"] == 1


def test_no_match_anywhere_yields_nan_origin(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(),  # totally empty BRENDA
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["kcat"] == 0.0
    assert pd.isna(row["origin"])
    assert pd.isna(row["wildcard_level"])


def test_force_wildcard_level_skips_exact_match(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "yeast", 5.0),  # exact match for kcat=5
            ("1.1.1.2", "anything", "yeast", 99.0),  # only wildcard match
        ]),
        _phyl_dist(["yeast"]),
        force_wildcard_level=1,
    )
    row = df.iloc[0]
    # With force_wildcard_level=1, we never try the exact 1.1.1.1 token;
    # we start at 1.1.1.- which matches both DB rows.
    # Substrate matching is disabled (wildcard); origin 3 (org-any-kcat).
    # Max kcat among all matches.
    assert row["wildcard_level"] == 1
    assert row["kcat"] == pytest.approx(99.0)


# --------------------------------------------------------------------------- #
# Multi-EC tokens per reaction
# --------------------------------------------------------------------------- #

def test_multi_ec_picks_token_with_minimum_wildcards(tmp_path):
    """Token A (1.1.1.1) matches at wc=1; token B (2.7.7.7) matches at wc=0.
    B should win."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "1.1.1.1;2.7.7.7", [("alpha", -1.0)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.5", "alpha", "yeast", 100.0),  # only via wildcard
            ("2.7.7.7", "alpha", "yeast", 41.0),   # exact
        ]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["wildcard_level"] == 0
    assert row["kcat"] == pytest.approx(41.0)
    assert row["origin"] == 1


def test_multi_ec_picks_max_kcat_when_tied_on_wc_and_origin(tmp_path):
    """Both tokens match at wc=0, origin=1; pick the larger kcat."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "1.1.1.1;2.7.7.7", [("alpha", -1.0)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "yeast", 43.0),
            ("2.7.7.7", "alpha", "yeast", 47.0),
        ]),
        _phyl_dist(["yeast"]),
    )
    assert df.iloc[0]["kcat"] == pytest.approx(47.0)


def test_multi_ec_org_sa_token_wins_over_any_org_kcat_token(tmp_path):
    """Token A (1.1.1.1) only has an org-SA row (origin 5); token B
    (2.7.7.7) only has an any-org no-substrate kcat (origin 4). Across
    tokens the earlier-tried level wins, as within one token: A's org-SA
    is selected even though B's kcat is larger."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "1.1.1.1;2.7.7.7", [("alpha", -1.0)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(
            kcat_rows=[("2.7.7.7", "different_substrate", "ecoli", 100.0)],
            sa_rows=[("1.1.1.1", "yeast", 23.0, 50.0)],
        ),
        _phyl_dist(["yeast", "ecoli"], dist=np.array([[0, 5], [5, 0]])),
    )
    row = df.iloc[0]
    assert row["origin"] == 5
    assert row["kcat"] == pytest.approx(23.0)


# --------------------------------------------------------------------------- #
# Phylogenetic distance
# --------------------------------------------------------------------------- #

def test_closest_organism_used_when_exact_org_misses(tmp_path):
    """No yeast row, but ecoli is closer to yeast than fugu in PhylDist;
    pick ecoli over fugu."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "ecoli", 53.0),
            ("1.1.1.1", "alpha", "fugu", 57.0),
        ]),
        _phyl_dist(
            ["yeast", "ecoli", "fugu"],
            dist=np.array([
                [0, 1, 100],
                [1, 0, 100],
                [100, 100, 0],
            ]),
        ),
    )
    row = df.iloc[0]
    assert row["kcat"] == pytest.approx(53.0)
    assert row["origin"] == 2


def test_genus_fallback_resolves_model_organism_via_kegg(tmp_path):
    """The model organism's exact name is absent from PhylDist, but its
    genus is present. The genus fallback resolves the model org to a
    KEGG index, which then drives phylogenetic-distance filtering
    against BRENDA rows. With BRENDA containing only a non-yeast
    organism, this hits origin 4 (any-org-no-subs-kcat)."""
    adapter = _adapter_with_org(tmp_path, "Foo qux")  # not in PhylDist directly
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "different", "Bar baz", 59.0),
        ]),
        _phyl_dist(
            ["Foo bar", "Bar baz"],  # `Foo bar` provides genus `foo` for fallback
            dist=np.array([[0, 2], [2, 0]]),
        ),
    )
    row = df.iloc[0]
    assert row["origin"] == 4
    assert row["kcat"] == pytest.approx(59.0)


# --------------------------------------------------------------------------- #
# Diffusion-limit cap
# --------------------------------------------------------------------------- #

def test_diffusion_limit_caps_huge_sa_kcat(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(sa_rows=[("1.1.1.1", "yeast", 1e9, 50.0)]),
        _phyl_dist(["yeast"]),
    )
    assert df.iloc[0]["kcat"] == pytest.approx(1e7)


# --------------------------------------------------------------------------- #
# Substrate-coefficient normalization
# --------------------------------------------------------------------------- #

def test_substrate_coefficient_normalizes_kcat(tmp_path):
    """A reaction with substrate coefficient 2 should divide the
    matched kcat by min(coeff)=2."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -2.0)])])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[("1.1.1.1", "alpha", "yeast", 60.0)]),
        _phyl_dist(["yeast"]),
    )
    assert df.iloc[0]["kcat"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# ec_rxns selector
# --------------------------------------------------------------------------- #

def test_ec_rxns_subset_only_processed(tmp_path):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "1.1.1.1", [("alpha", -1.0)]),
        ("r2", "2.7.7.7", [("alpha", -1.0)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[
            ("1.1.1.1", "alpha", "yeast", 10.0),
            ("2.7.7.7", "alpha", "yeast", 20.0),
        ]),
        _phyl_dist(["yeast"]),
        ec_rxns=["r1"],
    )
    assert list(df["rxn_id"]) == ["r1"]


# --------------------------------------------------------------------------- #
# Logger summary
# --------------------------------------------------------------------------- #

def test_summary_logged_when_matches_found(tmp_path, caplog):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    with caplog.at_level(logging.INFO):
        fuzzy_kcat_matching(
            model,
            _brenda(kcat_rows=[("1.1.1.1", "alpha", "yeast", 5.0)]),
            _phyl_dist(["yeast"]),
        )
    assert "matched 1 of 1" in caplog.text


def test_summary_logged_when_no_matches(tmp_path, caplog):
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    with caplog.at_level(logging.INFO):
        fuzzy_kcat_matching(
            model, _brenda(), _phyl_dist(["yeast"]),
        )
    assert "no matches" in caplog.text


# --------------------------------------------------------------------------- #
# Aggregation choice
# --------------------------------------------------------------------------- #

def test_aggregate_kcats_max_or_median():
    from geckopy.gather_kcats.fuzzy_kcat_matching._brenda_query import (
        _aggregate_kcats,
    )
    arr = np.array([1.0, 2.0, 3.0, 100.0])
    assert _aggregate_kcats(arr, "max") == 100.0
    assert _aggregate_kcats(arr, "median") == pytest.approx(2.5)


def _adapter_with_aggregate(
    tmp_path: Path, org_name: str, aggregate: str,
) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\n'
        f'org_name = "{org_name}"\n'
        f'kcat_aggregate_brenda = "{aggregate}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def test_aggregate_default_max_when_no_explicit_arg(tmp_path):
    """With the adapter at its default ``kcat_aggregate_brenda='max'``, an
    omitted ``aggregate`` arg picks the max of the matched BRENDA rows."""
    adapter = _adapter_with_org(tmp_path, "yeast")  # adapter default = max
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    brenda = _brenda([
        ("1.1.1.1", "alpha", "yeast", 2.0),
        ("1.1.1.1", "alpha", "yeast", 8.0),
        ("1.1.1.1", "alpha", "yeast", 100.0),
    ])
    df = fuzzy_kcat_matching(model, brenda, _phyl_dist(["yeast"]))
    assert df["kcat"].iloc[0] == 100.0


def test_aggregate_adapter_median_changes_default(tmp_path):
    """Adapter ``kcat_aggregate_brenda='median'`` flips the default
    behaviour to median without the caller passing ``aggregate=``."""
    adapter = _adapter_with_aggregate(tmp_path, "yeast", "median")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    brenda = _brenda([
        ("1.1.1.1", "alpha", "yeast", 2.0),
        ("1.1.1.1", "alpha", "yeast", 8.0),
        ("1.1.1.1", "alpha", "yeast", 100.0),
    ])
    df = fuzzy_kcat_matching(model, brenda, _phyl_dist(["yeast"]))
    assert df["kcat"].iloc[0] == 8.0


def test_explicit_aggregate_overrides_adapter(tmp_path):
    """Caller-passed ``aggregate=`` wins over the adapter setting."""
    adapter = _adapter_with_aggregate(tmp_path, "yeast", "median")
    model = _ec_model(adapter, [("r1", "1.1.1.1", [("alpha", -1.0)])])
    brenda = _brenda([
        ("1.1.1.1", "alpha", "yeast", 2.0),
        ("1.1.1.1", "alpha", "yeast", 8.0),
        ("1.1.1.1", "alpha", "yeast", 100.0),
    ])
    df = fuzzy_kcat_matching(
        model, brenda, _phyl_dist(["yeast"]), aggregate="max",
    )
    assert df["kcat"].iloc[0] == 100.0


def test_kcat_is_scaled_by_the_matched_substrate_not_the_smallest(tmp_path):
    """A BRENDA kcat is a turnover per molecule of the substrate it was
    measured on, so it converts to a reaction turnover by dividing by
    that substrate's coefficient.

    Cytochrome c oxidase is the case that matters: yeast-GEM writes it
    per one cytochrome c, so oxygen carries 0.25. Dividing BRENDA's
    1500/s by the smallest coefficient in the reaction reports 6000/s,
    four times the measured turnover and well past what the enzyme can
    do.
    """
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "7.1.1.9", [("ferrocytochrome_c", -1.0), ("oxygen", -0.25)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[("7.1.1.9", "ferrocytochrome_c", "yeast", 1500.0)]),
        _phyl_dist(["yeast"]),
    )
    row = df.iloc[0]
    assert row["origin"] == 1
    assert row["kcat"] == pytest.approx(1500.0)


def test_kcat_is_divided_when_the_matched_substrate_is_taken_more_than_once(tmp_path):
    """The division is still needed where it was meant to apply: two
    molecules of the measured substrate per turnover halve the rate."""
    adapter = _adapter_with_org(tmp_path, "yeast")
    model = _ec_model(adapter, [
        ("r1", "1.1.1.1", [("alpha", -2.0), ("beta", -1.0)]),
    ])
    df = fuzzy_kcat_matching(
        model,
        _brenda(kcat_rows=[("1.1.1.1", "alpha", "yeast", 100.0)]),
        _phyl_dist(["yeast"]),
    )
    assert df.iloc[0]["kcat"] == pytest.approx(50.0)
