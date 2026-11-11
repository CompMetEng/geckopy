"""Tests for apply_custom_kcats (legacy TSV-based wrapper)."""
import logging
from pathlib import Path

import cobra
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import apply_custom_kcats

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"

HEADER = "proteins\tgenes\tgene_name\tkcat\trxns\tnotes\tstoicho\n"


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


def _notes(model: EcModel, rxn_id: str) -> str:
    return model.ec.notes[model.ec.rxns.index(rxn_id)]


def _write_tsv(path: Path, rows: list[str]) -> None:
    path.write_text(HEADER + "".join(r + "\n" for r in rows))


# --------------------------------------------------------------------------- #
# Mode A: rxns only
# --------------------------------------------------------------------------- #

def test_mode_a_applies_to_unsuffixed_id(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t50\tR3\tnote_a\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R3") == 50.0
    assert _source(ec_model, "R3") == "custom"
    assert _notes(ec_model, "R3") == "note_a"


def test_mode_a_expands_to_all_isozymes(tmp_path):
    """R2 in mode A should match R2_EXP_1 and R2_EXP_2 but not R2_REV_*."""
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR2\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 10.0
    assert _kcat(ec_model, "R2_EXP_2") == 10.0
    assert _kcat(ec_model, "R2_REV_EXP_1") == 0
    assert _kcat(ec_model, "R2_REV_EXP_2") == 0


def test_mode_a_rev_direction_is_honored(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t20\tR2_REV\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 0
    assert _kcat(ec_model, "R2_REV_EXP_1") == 20.0
    assert _kcat(ec_model, "R2_REV_EXP_2") == 20.0


def test_mode_a_multiple_comma_separated_rxns(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t77\tR3, R5\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R3") == 77.0
    assert _kcat(ec_model, "R5") == 77.0


# --------------------------------------------------------------------------- #
# Mode B: proteins only
# --------------------------------------------------------------------------- #

def test_mode_b_single_protein_full_match(tmp_path):
    """P3 catalyzes R2_EXP_2 (alone) and R2_REV_EXP_2 (alone). Both
    full-match the singleton {P3}. Both should get the kcat."""
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P3\t\t\t200\t\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R2_EXP_2") == 200.0
    assert _kcat(ec_model, "R2_REV_EXP_2") == 200.0


def test_mode_b_protein_pair_full_match(tmp_path):
    """P1 + P2 full-matches reactions whose enzyme set is exactly {P1, P2}.
    R2_EXP_1 and R2_REV_EXP_1 are such reactions in ecTestGEM."""
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P1 + P2\t\t\t100\t\t\t1 + 1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 100.0
    assert _kcat(ec_model, "R2_REV_EXP_1") == 100.0


def test_mode_b_partial_match_logs_but_does_not_apply(tmp_path, caplog):
    """P1 alone is a candidate for R2_EXP_1 (50% match: {P1} vs {P1, P2}).
    Should log a partial-match warning and not apply."""
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P1\t\t\t999\t\t\t1"])

    with caplog.at_level(logging.WARNING):
        apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _kcat(ec_model, "R2_EXP_1") == 0
    assert "partial match" in caplog.text


def test_mode_b_unknown_protein_logs_warning(tmp_path, caplog):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P_NOT_REAL\t\t\t10\t\t\t1"])

    with caplog.at_level(logging.WARNING):
        apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert "P_NOT_REAL" in caplog.text


# --------------------------------------------------------------------------- #
# Mode C: proteins + rxns
# --------------------------------------------------------------------------- #

def test_mode_c_restricts_to_listed_rxns(tmp_path):
    """Mode C with P3 and rxns=R2_EXP_2 should hit only R2_EXP_2,
    not R2_REV_EXP_2 (which P3 also catalyzes)."""
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P3\t\t\t300\tR2\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    # R2 (no _REV) expands to R2_EXP_1 and R2_EXP_2. Only R2_EXP_2 is
    # catalyzed by P3 alone (full match).
    assert _kcat(ec_model, "R2_EXP_2") == 300.0
    assert _kcat(ec_model, "R2_REV_EXP_2") == 0


# --------------------------------------------------------------------------- #
# Notes appending
# --------------------------------------------------------------------------- #

def test_notes_append_to_existing(tmp_path):
    ec_model = _ectestgem_ec_model()
    # Pre-set a note manually.
    ec_model.ec.notes[ec_model.ec.rxns.index("R3")] = "pre-existing"

    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR3\tnew note\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert _notes(ec_model, "R3") == "pre-existing, new note"


def test_notes_overwrite_when_existing_empty(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR3\tnote\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)
    assert _notes(ec_model, "R3") == "note"


def test_empty_notes_column_does_not_change_existing(tmp_path):
    ec_model = _ectestgem_ec_model()
    ec_model.ec.notes[ec_model.ec.rxns.index("R3")] = "keep me"

    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR3\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)
    assert _notes(ec_model, "R3") == "keep me"


# --------------------------------------------------------------------------- #
# stoicho column
# --------------------------------------------------------------------------- #

def test_stoicho_warning_logged_for_nontrivial(tmp_path, caplog):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P1 + P2\t\t\t10\t\t\t3 + 2"])

    with caplog.at_level(logging.WARNING):
        apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert "stoichiometry" in caplog.text


def test_stoicho_trivial_no_warning(tmp_path, caplog):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["P1 + P2\t\t\t10\t\t\t1 + 1"])

    with caplog.at_level(logging.WARNING):
        apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert "stoichiometry" not in caplog.text


# --------------------------------------------------------------------------- #
# apply parameter and S matrix
# --------------------------------------------------------------------------- #

def test_apply_true_writes_to_s_matrix(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR3\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=True)

    r3 = ec_model.reactions.get_by_id("R3")
    coef = next((c for m, c in r3.metabolites.items() if m.id == "prot_P4"), 0.0)
    assert coef != 0.0


def test_apply_false_leaves_s_matrix_unchanged(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tR3\t\t1"])

    apply_custom_kcats(ec_model, path=tsv, apply=False)

    r3 = ec_model.reactions.get_by_id("R3")
    coef = next((c for m, c in r3.metabolites.items() if m.id == "prot_P4"), 0.0)
    assert coef == 0.0


# --------------------------------------------------------------------------- #
# File handling and errors
# --------------------------------------------------------------------------- #

def test_apply_custom_kcats_is_deprecated():
    ec_model = _ectestgem_ec_model()
    with pytest.warns(DeprecationWarning, match="convert_custom_kcats"):
        apply_custom_kcats(ec_model, apply=False)


def test_missing_file_raises(tmp_path):
    ec_model = _ectestgem_ec_model()
    with pytest.raises(FileNotFoundError):
        apply_custom_kcats(ec_model, path=tmp_path / "no_such.tsv")


def test_default_path_uses_adapter():
    """Without an explicit path, the function should look in the
    adapter's data folder. ecTestGEM ships a customKcats.tsv there."""
    ec_model = _ectestgem_ec_model()
    apply_custom_kcats(ec_model, apply=False)
    # The ecTestGEM customKcats.tsv has rows for P3, P1+P2, and R2_REV+R5.
    # We rely on at least one having taken effect.
    has_custom_source = any(s == "custom" for s in ec_model.ec.source)
    assert has_custom_source


def test_header_with_too_few_columns_raises(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "bad.tsv"
    tsv.write_text("only\ttwo\n")
    with pytest.raises(ValueError, match="header"):
        apply_custom_kcats(ec_model, path=tsv)


def test_no_matches_logs_summary(tmp_path, caplog):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    _write_tsv(tsv, ["\t\t\t10\tNOT_A_REACTION\t\t1"])

    with caplog.at_level(logging.WARNING):
        apply_custom_kcats(ec_model, path=tsv, apply=False)

    assert "no reactions were updated" in caplog.text or \
           "no reactions matched" in caplog.text


def test_blank_line_skipped(tmp_path):
    ec_model = _ectestgem_ec_model()
    tsv = tmp_path / "customKcats.tsv"
    tsv.write_text(
        HEADER + "\n" + "\t\t\t10\tR3\t\t1\n" + "\n"
    )
    apply_custom_kcats(ec_model, path=tsv, apply=False)
    assert _kcat(ec_model, "R3") == 10.0
