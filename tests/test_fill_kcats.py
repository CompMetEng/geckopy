"""Tests for fill_kcats_from_isozymes."""
import logging
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.ec_data import EcData
from geckopy.ec_model.pipeline import (
    fill_kcats_from_isozymes,
    set_kcat_for_reactions,
)


def _isozyme_model(kcats: list[float]) -> EcModel:
    """Minimal EcModel: ``len(kcats)`` isozymes of one base reaction
    ``R_EXP_<i>``; a 0 entry marks a missing kcat. apply=False keeps it
    independent of the cobra reactions."""
    n = len(kcats)
    model = EcModel("iso")
    model.ec = EcData(
        rxns=[f"R_EXP_{i + 1}" for i in range(n)],
        kcat=np.array(kcats, dtype=float),
        source=["x" if k else "" for k in kcats],
        notes=[""] * n,
        eccodes=[""] * n,
        rxn_enz_mat=sparse.csr_matrix((n, 0), dtype=float),
    )
    return model

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


# --------------------------------------------------------------------------- #
# Helpers
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


def _kcat(model: EcModel, rxn_id: str) -> float:
    return float(model.ec.kcat[model.ec.rxns.index(rxn_id)])


def _source(model: EcModel, rxn_id: str) -> str:
    return model.ec.source[model.ec.rxns.index(rxn_id)]


# --------------------------------------------------------------------------- #
# Basic isozyme averaging
# --------------------------------------------------------------------------- #

def test_fills_single_missing_isozyme_with_sibling_kcat():
    """R2_EXP_1 known, R2_EXP_2 missing; the missing one should get the
    same kcat as R2_EXP_1 (mean of one element is itself)."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 100.0
    assert _kcat(ec_model, "R2_EXP_2") == 100.0
    assert _source(ec_model, "R2_EXP_2") == "isozymes"


def test_fills_with_mean_of_multiple_known_siblings():
    """If R2_EXP_1=80 and R2_EXP_2 missing, fill with 80. If a third
    isozyme were known too, the mean would be used."""
    # ecTestGEM only has two isozymes for R2 in each direction, so
    # construct the case manually.
    ec_model = _ectestgem_ec_model()

    # R2 (forward) has R2_EXP_1 and R2_EXP_2. Set both, then unset one.
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 80.0, apply=False)
    set_kcat_for_reactions(ec_model, ["R2_EXP_2"], 120.0, apply=False)

    # R2_REV has R2_REV_EXP_1 and R2_REV_EXP_2. Leave R2_REV_EXP_2 NaN
    # and set R2_REV_EXP_1 = 50. The average of {50} is 50.
    set_kcat_for_reactions(ec_model, ["R2_REV_EXP_1"], 50.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R2_REV_EXP_2") == 50.0
    # The forward R2 kcats already had values, so they should be unchanged.
    assert _kcat(ec_model, "R2_EXP_1") == 80.0
    assert _kcat(ec_model, "R2_EXP_2") == 120.0


def test_aggregate_median_vs_mean():
    """Known siblings {1, 2, 100}; median fill = 2, mean fill = 34.33."""
    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    fill_kcats_from_isozymes(model, apply=False, aggregate="median")
    assert model.ec.kcat[3] == pytest.approx(2.0)

    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    fill_kcats_from_isozymes(model, apply=False)  # default "mean"
    assert model.ec.kcat[3] == pytest.approx((1.0 + 2.0 + 100.0) / 3.0)

    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    fill_kcats_from_isozymes(model, apply=False, aggregate="max")
    assert model.ec.kcat[3] == pytest.approx(100.0)


def test_aggregate_invalid_raises():
    model = _isozyme_model([1.0, 0.0])
    with pytest.raises(ValueError, match="aggregate"):
        fill_kcats_from_isozymes(model, apply=False, aggregate="nope")


def test_does_not_overwrite_known_kcats():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)
    set_kcat_for_reactions(ec_model, ["R2_EXP_2"], 200.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 100.0
    assert _kcat(ec_model, "R2_EXP_2") == 200.0
    # Neither got the "isozymes" source, since neither was filled.
    assert _source(ec_model, "R2_EXP_1") != "isozymes"
    assert _source(ec_model, "R2_EXP_2") != "isozymes"


# --------------------------------------------------------------------------- #
# Reverse direction is treated separately
# --------------------------------------------------------------------------- #

def test_rev_direction_is_not_a_sibling_of_forward():
    """R2_EXP_1 known; R2_REV_EXP_1 is NOT considered a sibling and
    should not be filled even though they share the underlying chemistry."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R2_REV_EXP_1") == 0
    assert _kcat(ec_model, "R2_REV_EXP_2") == 0


def test_rev_isozymes_share_among_themselves():
    """R2_REV_EXP_1 known should fill R2_REV_EXP_2 but not R2_EXP_*."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_REV_EXP_1"], 75.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R2_REV_EXP_2") == 75.0
    assert _kcat(ec_model, "R2_EXP_1") == 0
    assert _kcat(ec_model, "R2_EXP_2") == 0


# --------------------------------------------------------------------------- #
# Single-isozyme reactions
# --------------------------------------------------------------------------- #

def test_single_isozyme_reaction_stays_zero():
    """R3 has no siblings (it was never expanded; ec.rxns has just 'R3').
    With an unset kcat (0) and no siblings, it stays 0."""
    ec_model = _ectestgem_ec_model()
    # All unset, including R3. R3 has no _EXP_ siblings.
    fill_kcats_from_isozymes(ec_model, apply=False)
    assert _kcat(ec_model, "R3") == 0


def test_single_isozyme_does_not_fill_itself():
    """R3 with R3=10 should stay 10; the function should not invent siblings."""
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _kcat(ec_model, "R3") == 10.0
    assert _source(ec_model, "R3") != "isozymes"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_all_unset_kcats_does_nothing(caplog):
    ec_model = _ectestgem_ec_model()
    with caplog.at_level(logging.WARNING):
        fill_kcats_from_isozymes(ec_model, apply=False)
    assert "no known values" in caplog.text
    assert (ec_model.ec.kcat == 0).all()


def test_no_missing_to_fill_logs_info(caplog):
    """If every kcat is set, the function should be a quiet no-op
    (or info-level log)."""
    ec_model = _ectestgem_ec_model()
    ec_model.ec.kcat[:] = 50.0  # all set

    with caplog.at_level(logging.INFO):
        fill_kcats_from_isozymes(ec_model, apply=False)

    assert np.all(ec_model.ec.kcat == 50.0)


def test_gecko_light_raises():
    ec_model = _ectestgem_ec_model()
    ec_model.ec.gecko_light = True
    with pytest.raises(NotImplementedError, match="gecko-light"):
        fill_kcats_from_isozymes(ec_model)


def test_empty_ec_model():
    """A freshly-constructed EcModel (no make_ec_model run) should not crash."""
    ec_model = EcModel("empty")
    # ec.kcat is shape (0,), nothing to iterate.
    fill_kcats_from_isozymes(ec_model, apply=False)
    assert ec_model.ec.kcat.shape == (0,)


# --------------------------------------------------------------------------- #
# apply parameter and S matrix
# --------------------------------------------------------------------------- #

def test_apply_true_writes_to_s_matrix():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=True)

    r = ec_model.reactions.get_by_id("R2_EXP_2")
    coef_p3 = next(
        (c for m, c in r.metabolites.items() if m.id == "prot_P3"), 0.0
    )
    assert coef_p3 != 0.0


def test_apply_false_leaves_s_matrix_unchanged():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    r = ec_model.reactions.get_by_id("R2_EXP_2")
    coef_p3 = next(
        (c for m, c in r.metabolites.items() if m.id == "prot_P3"), 0.0
    )
    assert coef_p3 == 0.0


# --------------------------------------------------------------------------- #
# Source attribution
# --------------------------------------------------------------------------- #

def test_filled_kcats_have_isozymes_source():
    ec_model = _ectestgem_ec_model()
    set_kcat_for_reactions(ec_model, ["R2_EXP_1"], 100.0, apply=False)

    fill_kcats_from_isozymes(ec_model, apply=False)

    assert _source(ec_model, "R2_EXP_2") == "isozymes"
    # The source of the originally-set reaction should be untouched.
    assert _source(ec_model, "R2_EXP_1") == "manual"


# --------------------------------------------------------------------------- #
# Adapter-default aggregate (params.kcat_aggregate_isozymes)
# --------------------------------------------------------------------------- #

def _adapter_with_isozyme_aggregate(
    tmp_path: Path, aggregate: str | None = None,
) -> ModelAdapter:
    body = ['conv_gem = "dummy.xml"', 'org_name = "yeast"']
    if aggregate is not None:
        body.append(f'kcat_aggregate_isozymes = "{aggregate}"')
    (tmp_path / "model_adapter.toml").write_text("\n".join(body) + "\n")
    return ModelAdapter.from_folder(tmp_path)


def test_aggregate_default_mean_with_adapter_default(tmp_path):
    """Adapter defaults to ``kcat_aggregate_isozymes='mean'``; omitting
    ``aggregate`` falls through to that default."""
    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    model.adapter = _adapter_with_isozyme_aggregate(tmp_path)
    fill_kcats_from_isozymes(model, apply=False)
    assert model.ec.kcat[3] == pytest.approx((1.0 + 2.0 + 100.0) / 3.0)


def test_aggregate_adapter_median_flips_default(tmp_path):
    """With the adapter at ``kcat_aggregate_isozymes='median'``, an
    omitted ``aggregate`` falls through to median averaging."""
    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    model.adapter = _adapter_with_isozyme_aggregate(tmp_path, "median")
    fill_kcats_from_isozymes(model, apply=False)
    assert model.ec.kcat[3] == pytest.approx(2.0)


def test_aggregate_default_mean_when_no_adapter():
    """Without an adapter attached, the mean default is used."""
    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    fill_kcats_from_isozymes(model, apply=False)
    assert model.ec.kcat[3] == pytest.approx((1.0 + 2.0 + 100.0) / 3.0)


def test_explicit_aggregate_overrides_adapter(tmp_path):
    """Caller-passed ``aggregate=`` wins over the adapter setting."""
    model = _isozyme_model([1.0, 2.0, 100.0, 0.0])
    model.adapter = _adapter_with_isozyme_aggregate(tmp_path, "median")
    fill_kcats_from_isozymes(model, apply=False, aggregate="max")
    assert model.ec.kcat[3] == pytest.approx(100.0)
