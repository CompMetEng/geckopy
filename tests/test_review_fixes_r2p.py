"""get_conc_control_coeffs: the verification rule of the reduced-cost path.

A reduced cost is kept when raising the usage bound by 0.01 percent raises the optimum by at
least half of the rise the reduced cost predicts and by at least 1e-12 times max(1, |optimum|).
Every nonzero reduced cost is checked, however small. A usage reaction held at its lower bound
is not checked. The check solve is replaced by a stub that reports a chosen rise, so each case
fixes the realised gain exactly.
"""
import sys
from types import SimpleNamespace

import pytest

from geckopy.limit_proteins import get_conc_control_coeffs

from test_get_conc_control_coeffs import _build_toy_ec_model

SOLVERS = ["glpk", "glpk_exact", "gurobi"]
MODULE = "geckopy.limit_proteins.get_conc_control_coeffs"


def _with_solver(model, solver):
    try:
        model.solver = solver
        model.optimize()
    except Exception as exc:  # solver not installed or not licensed
        pytest.skip(f"solver {solver} unavailable: {exc}")
    return model


def _stub_check_solve(monkeypatch, model, gain):
    """The first solve is real; later solves report the first optimum plus ``gain``."""
    real = model.optimize
    state = {"calls": 0, "initial": None}

    def optimize(*args, **kwargs):
        sol = real(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == 1:
            state["initial"] = sol.objective_value
            return sol
        return SimpleNamespace(
            status="optimal", objective_value=state["initial"] + gain, fluxes=sol.fluxes
        )

    monkeypatch.setattr(model, "optimize", optimize, raising=False)
    return state


def _fixed_reduced_cost(monkeypatch, value):
    module = sys.modules[MODULE]
    monkeypatch.setattr(module, "_forward_reduced_cost", lambda rxn: value)


# The toy with one enzyme of bound 5 and one unit of enzyme per unit of flux has reduced cost 1,
# so the predicted gain of the 0.01 percent step is 1 * 5 * 1e-4 = 5e-4.

@pytest.mark.parametrize("solver", SOLVERS)
def test_a_gain_below_half_the_prediction_gives_zero(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _stub_check_solve(monkeypatch, model, 0.3 * 5e-4)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_gain_above_half_the_prediction_keeps_the_reduced_cost(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _stub_check_solve(monkeypatch, model, 0.6 * 5e-4)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_gain_below_the_floor_is_noise(solver, monkeypatch):
    # Reduced cost 1e-8 on a bound of 5 predicts 5e-12; the floor is 1e-12 * max(1, 5) = 5e-12.
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _fixed_reduced_cost(monkeypatch, 1e-8)
    _stub_check_solve(monkeypatch, model, 4e-12)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_gain_above_the_floor_is_kept(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _fixed_reduced_cost(monkeypatch, 1e-8)
    _stub_check_solve(monkeypatch, model, 6e-12)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == pytest.approx(1e-8)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_reduced_cost_below_1e_10_is_checked_and_kept(solver, monkeypatch):
    # Reduced cost 5e-11 on a bound of 500 predicts 2.5e-12, above the floor of 1e-12.
    model = _with_solver(_build_toy_ec_model(enzyme_ub=500.0, enzyme_coeff_in_R=-2000.0), solver)
    _fixed_reduced_cost(monkeypatch, 5e-11)
    _stub_check_solve(monkeypatch, model, 2.5e-12)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == pytest.approx(5e-11)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_zero_reduced_cost_is_not_checked(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _fixed_reduced_cost(monkeypatch, 0.0)
    state = _stub_check_solve(monkeypatch, model, 1.0)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == 0.0
    assert state["calls"] == 1


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_usage_reaction_held_at_its_lower_bound_is_not_checked(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    usage = model.reactions.get_by_id("usage_prot_E")
    usage.lower_bound = 2.0
    usage.upper_bound = 2.0
    _fixed_reduced_cost(monkeypatch, 1.0)
    state = _stub_check_solve(monkeypatch, model, 1.0)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == 0.0
    assert state["calls"] == 1
