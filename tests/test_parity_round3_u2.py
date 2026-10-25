"""An empty update list in apply_kcat_constraints, and a quoted reaction list in
apply_custom_kcats, as in GECKO."""
import copy
from pathlib import Path
from types import SimpleNamespace

import cobra
import numpy as np
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import (
    apply_custom_kcats,
    apply_kcat_constraints,
    set_kcat_for_reactions,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
HEADER = "proteins\tgenes\tgene_name\tkcat\trxns\tnotes\tstoicho\n"

_CACHE: dict[bool, EcModel] = {}


def _model(light: bool = False) -> EcModel:
    if light not in _CACHE:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _CACHE[light] = make_ec_model(cobra_model, adapter, gecko_light=light)
    return copy.deepcopy(_CACHE[light])


def _coef(model: EcModel, rxn_id: str, met_id: str) -> float:
    for met, coef in model.reactions.get_by_id(rxn_id).metabolites.items():
        if met.id == met_id:
            return coef
    return 0.0


@pytest.mark.parametrize("empty", [[], (), np.array([], dtype=str)])
def test_empty_update_list_updates_all_reactions(empty):
    model = _model()
    model.ec.kcat[model.ec.rxns.index("R3")] = 10.0
    assert _coef(model, "R3", "prot_P4") == 0.0

    apply_kcat_constraints(model, update_rxns=empty)

    assert _coef(model, "R3", "prot_P4") < 0.0


def test_empty_update_list_updates_all_reactions_light():
    model = _model(light=True)
    model.ec.kcat[model.ec.rxns.index("001_R3")] = 10.0
    assert _coef(model, "R3", "prot_pool") == 0.0

    apply_kcat_constraints(model, update_rxns=[])

    assert _coef(model, "R3", "prot_pool") < 0.0


def test_a_list_naming_one_reaction_still_updates_only_that_one():
    model = _model()
    model.ec.kcat[model.ec.rxns.index("R3")] = 10.0
    model.ec.kcat[model.ec.rxns.index("R5")] = 10.0

    apply_kcat_constraints(model, update_rxns=["R3"])

    assert _coef(model, "R3", "prot_P4") < 0.0
    assert _coef(model, "R5", "prot_P5") == 0.0


def test_set_kcat_for_reactions_without_ids_does_not_rewrite_the_model():
    model = _model()
    model.ec.kcat[model.ec.rxns.index("R3")] = 10.0

    updated = set_kcat_for_reactions(model, [], 5.0, apply=True)

    assert updated == []
    assert _coef(model, "R3", "prot_P4") == 0.0


def test_evotune_scoring_skips_the_constraint_update_for_no_tunable_rows(monkeypatch):
    from geckopy.kcat_tuning.evotune import tuning

    calls = []

    class _Stop(Exception):
        pass

    def _record(model, update_rxns=None):
        calls.append(update_rxns)

    def _stop(model):
        raise _Stop

    monkeypatch.setattr(tuning, "apply_kcat_constraints", _record)
    monkeypatch.setattr(tuning, "_reset_solver_basis", _stop)
    model = SimpleNamespace(ec=SimpleNamespace(kcat=np.zeros(3)))

    with pytest.raises(_Stop):
        tuning._score_kcat_vector(
            model, np.array([], dtype=int), [], None, {}, "bio",
            np.array([]), make_anaerobic=None, change_protein_biomass=None,
        )

    assert calls == []


def test_quoted_reaction_list_matches_the_listed_reactions(tmp_path):
    model = _model()
    tsv = tmp_path / "customKcats.tsv"
    tsv.write_text(HEADER + '\t\t\t50\t"R3,R5"\tquoted\t1\n')

    apply_custom_kcats(model, path=tsv, apply=False)

    for rxn in ("R3", "R5"):
        assert model.ec.kcat[model.ec.rxns.index(rxn)] == 50.0
        assert model.ec.source[model.ec.rxns.index(rxn)] == "custom"


def test_unquoted_and_single_reaction_lists_are_unchanged(tmp_path):
    model = _model()
    tsv = tmp_path / "customKcats.tsv"
    tsv.write_text(HEADER + "\t\t\t50\tR3,R5\tplain\t1\n")

    apply_custom_kcats(model, path=tsv, apply=False)

    assert model.ec.kcat[model.ec.rxns.index("R3")] == 50.0
    assert model.ec.kcat[model.ec.rxns.index("R5")] == 50.0
