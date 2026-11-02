"""Reduced costs without usable duals in flexibilize_enz_concs, and the positional kcat rule of
set_kcat_for_reactions when an input id matches nothing."""
import copy
import importlib
from pathlib import Path

import cobra
import numpy as np
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import set_kcat_for_reactions
from geckopy.limit_proteins.flexibilize_enz_concs import _unit_scores

from test_conc_control_coeffs_shadow import _with_solver
from test_flexibilize_enz_concs import _adapter, _build_toy

gcc = importlib.import_module("geckopy.limit_proteins.get_conc_control_coeffs")

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
_CACHE = {}


def _ectestgem():
    if "m" not in _CACHE:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        _CACHE["m"] = make_ec_model(cobra_model, adapter)
    return copy.deepcopy(_CACHE["m"])


def _scipy_toy(tmp_path):
    model = _with_solver(_build_toy(_adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass"), enzyme_ub=1.0, measured_conc=1.0), "scipy")
    reduced = model.optimize().reduced_costs
    if not np.isnan(np.asarray(reduced, dtype=float)).all():
        pytest.skip("this cobra returns usable reduced costs with the scipy solver")
    return model


def _optimal_in_sequence(monkeypatch, answers):
    calls = iter(answers)
    real = gcc._solution_is_optimal
    monkeypatch.setattr(gcc, "_solution_is_optimal", lambda sol: next(calls, real(sol)))


def test_unit_scores_are_zero_not_nan_when_the_lifted_pool_solve_is_not_optimal(tmp_path, monkeypatch):
    model = _scipy_toy(tmp_path)
    _optimal_in_sequence(monkeypatch, [True, False])
    scores = _unit_scores(model, ["E"], [[0]], 1.1, 0.75)
    assert not np.isnan(scores).any()
    assert (scores == 0.0).all()


def test_unit_scores_are_zero_not_nan_when_a_probe_is_not_optimal(tmp_path, monkeypatch):
    model = _scipy_toy(tmp_path)
    _optimal_in_sequence(monkeypatch, [True, True, False])
    scores = _unit_scores(model, ["E"], [[0]], 1.1, 0.75)
    assert not np.isnan(scores).any()
    assert (scores == 0.0).all()


def test_unit_scores_with_nan_reduced_costs_fall_back_to_the_probes(tmp_path):
    model = _scipy_toy(tmp_path)
    scores = _unit_scores(model, ["E"], [[0]], 1.1, 0.75)
    assert not np.isnan(scores).any()
    assert scores[0] > 0.0


def test_a_known_id_next_to_an_unknown_id_takes_the_values_by_position():
    model = _ectestgem()
    rows = [i for i, r in enumerate(model.ec.rxns) if r.split("_EXP_")[0] == "R2"]
    assert len(rows) == 2
    updated = set_kcat_for_reactions(model, ["R2", "nonsense"], [1.0, 2.0])
    assert sorted(updated) == sorted(model.ec.rxns[i] for i in rows)
    assert [float(model.ec.kcat[i]) for i in rows] == [1.0, 2.0]


def test_one_value_per_input_id_still_broadcasts_when_every_id_matches():
    model = _ectestgem()
    set_kcat_for_reactions(model, ["R2", "R3"], [7.0, 9.0])
    r2 = [float(model.ec.kcat[i]) for i, r in enumerate(model.ec.rxns) if r.split("_EXP_")[0] == "R2"]
    assert r2 == [7.0, 7.0]
    assert float(model.ec.kcat[model.ec.rxns.index("R3")]) == 9.0
