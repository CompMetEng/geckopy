"""Tests for set_kcat_for_reactions."""
from pathlib import Path

import cobra
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import (
    apply_kcat_constraints,
    set_kcat_for_reactions,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_ECTESTGEM_CACHE: EcModel | None = None


def _ectestgem_ec_model() -> EcModel:
    """Cached build of the ecTestGEM ecModel; deep-copied per call."""
    import copy as _copy
    global _ECTESTGEM_CACHE
    if _ECTESTGEM_CACHE is None:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _ECTESTGEM_CACHE = make_ec_model(cobra_model, adapter)
    return _copy.deepcopy(_ECTESTGEM_CACHE)


def _kcat_at(model: EcModel, rxn_id: str) -> float:
    return float(model.ec.kcat[model.ec.rxns.index(rxn_id)])


def _source_at(model: EcModel, rxn_id: str) -> str:
    return model.ec.source[model.ec.rxns.index(rxn_id)]


def _get_s_coef(rxn: cobra.Reaction, met_id: str) -> float:
    for m, c in rxn.metabolites.items():
        if m.id == met_id:
            return c
    return 0.0


# --------------------------------------------------------------------------- #
# Single suffixed ID
# --------------------------------------------------------------------------- #

def test_set_single_suffixed_rxn():
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=False)
    assert updated == ["R3"]
    assert _kcat_at(ec_model, "R3") == 10.0
    assert _source_at(ec_model, "R3") == "setKcatForReactions"


def test_set_single_kcat_scalar_only_form():
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(ec_model, ["R3"], 25.0, apply=False)
    assert updated == ["R3"]
    assert _kcat_at(ec_model, "R3") == 25.0


# --------------------------------------------------------------------------- #
# Un-suffixed ID expands to all isozymes
# --------------------------------------------------------------------------- #

def test_unsuffixed_rxn_expands_to_all_isozymes():
    """An un-suffixed base name matches only the ec.rxns entries whose ID
    reduces to that exact base after stripping a trailing `_EXP_<n>`: "R2"
    matches "R2_EXP_1"/"R2_EXP_2" but not "R2_REV_EXP_1"/"R2_REV_EXP_2",
    whose base is "R2_REV"."""
    ec_model = _ectestgem_ec_model()

    # Check what ec.rxns contains starting with R2.
    r2_entries = [r for r in ec_model.ec.rxns if r.startswith("R2")]
    # Expected: R2_EXP_1, R2_EXP_2, R2_REV_EXP_1, R2_REV_EXP_2.
    # All have the base name R2 after stripping _EXP_<n>.
    assert sorted(r2_entries) == sorted([
        "R2_EXP_1", "R2_EXP_2", "R2_REV_EXP_1", "R2_REV_EXP_2"
    ])

    # The base name "R2" should match all four, because stripping
    # _EXP_<n> from R2_EXP_1 gives "R2" but stripping from R2_REV_EXP_1
    # gives "R2_REV". So actually only R2_EXP_1 and R2_EXP_2 match "R2".
    updated = set_kcat_for_reactions(ec_model, ["R2"], 5.0, apply=False)
    assert sorted(updated) == ["R2_EXP_1", "R2_EXP_2"]
    assert _kcat_at(ec_model, "R2_EXP_1") == 5.0
    assert _kcat_at(ec_model, "R2_EXP_2") == 5.0
    # R2_REV_EXP_1 should NOT have been updated.
    assert _kcat_at(ec_model, "R2_REV_EXP_1") == 0


def test_unsuffixed_rxn_with_scalar_kcat_broadcasts():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2"], 7.5, apply=False)
    assert _kcat_at(ec_model, "R2_EXP_1") == 7.5
    assert _kcat_at(ec_model, "R2_EXP_2") == 7.5


def test_unsuffixed_rxn_with_list_kcat_assigns_positionally():
    """raven-gecko-parity#77: an un-suffixed ID expanding to multiple
    matches now accepts a length-N kcat list, assigning positionally in
    ec.rxns order -- matching MATLAB setKcatForReactions exactly, not
    refusing it."""
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(ec_model, ["R2"], [1.0, 2.0], apply=False)
    assert updated == ["R2_EXP_1", "R2_EXP_2"]
    assert _kcat_at(ec_model, "R2_EXP_1") == 1.0
    assert _kcat_at(ec_model, "R2_EXP_2") == 2.0


def test_set_specific_isozyme_with_suffix():
    """Passing R2_EXP_1 should match only that one, not its sibling."""
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(
        ec_model, ["R2_EXP_1"], 99.0, apply=False
    )
    assert updated == ["R2_EXP_1"]
    assert _kcat_at(ec_model, "R2_EXP_1") == 99.0
    assert _kcat_at(ec_model, "R2_EXP_2") == 0


# --------------------------------------------------------------------------- #
# Multiple input IDs
# --------------------------------------------------------------------------- #

def test_multiple_ids_with_scalar_kcat():
    """Scalar kcat broadcasts to all matched reactions across all inputs."""
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(
        ec_model, ["R3", "R5"], 12.0, apply=False
    )
    assert sorted(updated) == ["R3", "R5"]
    assert _kcat_at(ec_model, "R3") == 12.0
    assert _kcat_at(ec_model, "R5") == 12.0


def test_multiple_ids_with_per_input_kcat_list():
    """A list whose length equals len(rxn_ids) gives one value per input,
    broadcasting if that input expanded to multiple matches."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(
        ec_model,
        ["R3", "R5"],
        [10.0, 20.0],
        apply=False,
    )
    assert _kcat_at(ec_model, "R3") == 10.0
    assert _kcat_at(ec_model, "R5") == 20.0


def test_multiple_ids_with_per_match_kcat_list_when_all_unique():
    """When every input matches exactly one reaction, a length-N kcat
    where N == total_matches is also accepted."""
    ec_model = _ectestgem_ec_model()
    # Both R3 and R5 are 1:1, so per-input == per-match.
    set_kcat_for_reactions(
        ec_model,
        ["R3", "R5"],
        [10.0, 20.0],
        apply=False,
    )
    assert _kcat_at(ec_model, "R3") == 10.0
    assert _kcat_at(ec_model, "R5") == 20.0


def test_kcat_list_wrong_length_raises():
    ec_model = _ectestgem_ec_model()
    with pytest.raises(ValueError, match="kcat has length 3"):
        set_kcat_for_reactions(
            ec_model, ["R3", "R5"], [1.0, 2.0, 3.0], apply=False,
        )


# --------------------------------------------------------------------------- #
# Unknown ID, empty input
# --------------------------------------------------------------------------- #

def test_unknown_rxn_id_is_ignored():
    ec_model = _ectestgem_ec_model()
    before = list(ec_model.ec.kcat)
    assert set_kcat_for_reactions(ec_model, ["nonsense"], 1.0) == []
    assert list(ec_model.ec.kcat) == before


def test_unknown_with_known_changes_the_known_one():
    """An unknown ID is ignored; the known one is still set."""
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(ec_model, ["R3", "nonsense"], 1.0)
    assert updated == ["R3"]
    assert _kcat_at(ec_model, "R3") == 1.0


def test_ids_match_without_regard_to_letter_case():
    """The ``_EXP_<n>`` suffix itself is recognised in capitals only, as in
    MATLAB; the rest of the ID is compared without regard to case."""
    ec_model = _ectestgem_ec_model()
    updated = set_kcat_for_reactions(ec_model, ["r3"], 7.0)
    assert updated == ["R3"]
    assert _kcat_at(ec_model, "R3") == 7.0
    updated = set_kcat_for_reactions(ec_model, ["r2_EXP_1"], 8.0)
    assert updated == ["R2_EXP_1"]
    assert _kcat_at(ec_model, "R2_EXP_1") == 8.0


def test_empty_rxn_ids_returns_empty():
    ec_model = _ectestgem_ec_model()
    assert set_kcat_for_reactions(ec_model, [], 1.0) == []


# --------------------------------------------------------------------------- #
# apply=True propagates to S matrix
# --------------------------------------------------------------------------- #

def test_apply_true_writes_to_s_matrix():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=True)

    r3 = ec_model.reactions.get_by_id("R3")
    coef = _get_s_coef(r3, "prot_P4")
    expected = -(40000.0 / (10.0 * 3600.0))
    assert coef == pytest.approx(expected)


def test_apply_false_leaves_s_matrix_unchanged():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=False)

    r3 = ec_model.reactions.get_by_id("R3")
    assert _get_s_coef(r3, "prot_P4") == 0.0


def test_apply_default_is_false():
    """As MATLAB setKcatForReactions, only ec.kcat and ec.source change by
    default; the S matrix follows once apply_kcat_constraints runs."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0)  # no apply arg
    r3 = ec_model.reactions.get_by_id("R3")
    assert _get_s_coef(r3, "prot_P4") == 0.0
    assert _kcat_at(ec_model, "R3") == 10.0


def test_apply_only_updates_passed_reactions():
    """Updating one reaction's kcat leaves other reactions' S-matrix
    coefficients unchanged."""
    ec_model = _ectestgem_ec_model()
    # Pre-set R5 and apply manually.
    ec_model.ec.kcat[ec_model.ec.rxns.index("R5")] = 50.0
    apply_kcat_constraints(ec_model)
    r5_pre = _get_s_coef(ec_model.reactions.get_by_id("R5"), "prot_P5")
    assert r5_pre != 0.0

    # Now use set_kcat_for_reactions to update R3. R5 must be untouched.
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=True)
    r5_post = _get_s_coef(ec_model.reactions.get_by_id("R5"), "prot_P5")
    assert r5_post == r5_pre


# --------------------------------------------------------------------------- #
# Source string
# --------------------------------------------------------------------------- #

def test_source_string_matches_matlab():
    """raven-gecko-parity#77: ec.source reads 'setKcatForReactions',
    matching MATLAB's actual assignment (its docstring says 'from
    setKcatForReactions', but the real code writes the bare name)."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=False)
    assert _source_at(ec_model, "R3") == "setKcatForReactions"


# --------------------------------------------------------------------------- #
# gecko-light layout
# --------------------------------------------------------------------------- #

_ECTESTGEM_LIGHT_CACHE: EcModel | None = None


def _ectestgem_light_ec_model() -> EcModel:
    """Cached gecko-light build of ecTestGEM; deep-copied per call."""
    import copy as _copy
    global _ECTESTGEM_LIGHT_CACHE
    if _ECTESTGEM_LIGHT_CACHE is None:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _ECTESTGEM_LIGHT_CACHE = make_ec_model(
            cobra_model, adapter, gecko_light=True,
        )
    return _copy.deepcopy(_ECTESTGEM_LIGHT_CACHE)


def test_light_set_single_explicit_prefixed_id():
    """Passing ``001_R3`` should match only that one ec row."""
    ec = _ectestgem_light_ec_model()
    updated = set_kcat_for_reactions(ec, ["001_R3"], 10.0, apply=False)
    assert updated == ["001_R3"]
    assert _kcat_at(ec, "001_R3") == 10.0
    assert _source_at(ec, "001_R3") == "setKcatForReactions"


def test_light_base_id_matches_nothing():
    """The ``###_`` prefix is part of a light ec.rxns ID, as in MATLAB: the
    bare cobra ID ``R2`` matches none of ``001_R2`` and ``002_R2``."""
    ec = _ectestgem_light_ec_model()
    updated = set_kcat_for_reactions(ec, ["R2"], 5.0)
    assert updated == []
    assert _kcat_at(ec, "001_R2") == 0.0
    assert _kcat_at(ec, "002_R2") == 0.0


def test_light_both_prefixed_ids_set_both_isozyme_rows():
    ec = _ectestgem_light_ec_model()
    updated = set_kcat_for_reactions(ec, ["001_R2", "002_R2"], 5.0)
    assert updated == ["001_R2", "002_R2"]
    assert _kcat_at(ec, "001_R2") == 5.0
    assert _kcat_at(ec, "002_R2") == 5.0
    assert _kcat_at(ec, "001_R2_REV") == 0.0
    assert _kcat_at(ec, "002_R2_REV") == 0.0


def test_light_prefixed_ids_with_list_kcat():
    ec = _ectestgem_light_ec_model()
    updated = set_kcat_for_reactions(ec, ["001_R2", "002_R2"], [1.0, 2.0])
    assert updated == ["001_R2", "002_R2"]
    assert _kcat_at(ec, "001_R2") == 1.0
    assert _kcat_at(ec, "002_R2") == 2.0


def test_light_set_specific_isozyme_with_prefix():
    """Explicit ``002_R2`` should leave its sibling ``001_R2`` untouched."""
    ec = _ectestgem_light_ec_model()
    updated = set_kcat_for_reactions(ec, ["002_R2"], 99.0, apply=False)
    assert updated == ["002_R2"]
    assert _kcat_at(ec, "002_R2") == 99.0
    assert _kcat_at(ec, "001_R2") == 0.0


def test_light_apply_true_writes_prot_pool_coefficient():
    """With apply=True on a single-isozyme reaction, the chosen kcat must
    appear as a ``prot_pool`` coefficient on the cobra reaction."""
    ec = _ectestgem_light_ec_model()
    set_kcat_for_reactions(ec, ["001_R3"], 10.0, apply=True)

    r3 = ec.reactions.get_by_id("R3")
    # R3 is G4 alone (MW = 40000).
    coef = _get_s_coef(r3, "prot_pool")
    expected = -(40000.0 / (10.0 * 3600.0))
    assert coef == pytest.approx(expected)


def test_light_apply_picks_lowest_cost_isozyme():
    """If both isozymes of R2 get a kcat, apply_kcat must pick the
    cheapest (lowest MW_sum/kcat) one."""
    ec = _ectestgem_light_ec_model()
    set_kcat_for_reactions(ec, ["001_R2", "002_R2"], 5.0, apply=True)

    r2 = ec.reactions.get_by_id("R2")
    coef = _get_s_coef(r2, "prot_pool")
    # 001_R2 = G1+G2 (MW = 30000); 002_R2 = G3 alone (MW = 30000).
    # Same kcat and same MW_sum, so both have identical cost. The
    # implementation breaks ties on first-wins, so MW_sum = 30000.
    expected = -(30000.0 / (5.0 * 3600.0))
    assert coef == pytest.approx(expected)


def test_light_unknown_rxn_id_is_ignored():
    ec = _ectestgem_light_ec_model()
    assert set_kcat_for_reactions(ec, ["nonsense"], 1.0) == []
