"""get_conc_control_coeffs: reduced-cost path across solvers, and its fallbacks.

The coefficient of an enzyme is the reduced cost of its usage reaction's forward variable,
kept only where a 0.01 percent rise of the bound raises the optimum. An enzyme whose own
bound does not bind has coefficient 0 also while the shared protein pool binds.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from geckopy.limit_proteins import get_conc_control_coeffs
from geckopy.limit_proteins.get_conc_control_coeffs import (
    _finite_difference_coeffs,
    _forward_reduced_cost,
)

from test_get_conc_control_coeffs import _build_toy_ec_model

# The toy models have finite bounds only. glpk_exact aborts the process on a model with an
# infinite bound, which every ecModel built by geckopy has, whatever this function does.
SOLVERS = ["glpk", "glpk_exact", "gurobi", "scipy"]


def _with_solver(model, solver):
    try:
        model.solver = solver
        model.optimize()
    except Exception as exc:  # solver not installed or not licensed
        pytest.skip(f"solver {solver} unavailable: {exc}")
    return model


def _pool_binding_model(solver):
    """Enzyme bound 5 does not bind; the protein pool caps growth at 3."""
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    model.reactions.get_by_id("prot_pool_exchange").upper_bound = 3.0
    return model


def _degenerate_model(solver):
    """The enzyme bound and R_AB's own bound both equal 5: raising the enzyme bound alone raises nothing."""
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    model.reactions.get_by_id("R_AB").upper_bound = 5.0
    return model


@pytest.mark.parametrize("solver", SOLVERS)
def test_pool_binding_gives_zero_for_an_enzyme_whose_bound_does_not_bind(solver):
    model = _pool_binding_model(solver)
    assert model.optimize().objective_value == pytest.approx(3.0)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_binding_enzyme_bound_gives_its_marginal_growth(solver):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_shadow_price_equals_the_local_finite_difference(solver):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0, enzyme_coeff_in_R=-2.0), solver)
    _, coeffs = get_conc_control_coeffs(model)
    _, local = _finite_difference_coeffs(model, ["E"], 1.001, 0.0)
    assert coeffs[0] == pytest.approx(local[0], rel=1e-6)
    assert coeffs[0] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_degenerate_vertex_gives_zero(solver):
    model = _degenerate_model(solver)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_model_is_unchanged_after_the_call(solver):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    usage = model.reactions.get_by_id("usage_prot_E")
    before = (usage.lower_bound, usage.upper_bound)
    get_conc_control_coeffs(model)
    assert (usage.lower_bound, usage.upper_bound) == before


# --------------------------------------------------------------------------- #
# Duals that are missing or not finite
# --------------------------------------------------------------------------- #

class _Raises:
    @property
    def dual(self):
        raise NotImplementedError


def test_forward_reduced_cost_is_none_without_a_usable_dual():
    assert _forward_reduced_cost(SimpleNamespace(forward_variable=_Raises())) is None
    assert _forward_reduced_cost(SimpleNamespace(forward_variable=SimpleNamespace(dual=float("nan")))) is None
    assert _forward_reduced_cost(SimpleNamespace(forward_variable=SimpleNamespace(dual=float("inf")))) is None


def test_forward_reduced_cost_is_the_absolute_value():
    assert _forward_reduced_cost(SimpleNamespace(forward_variable=SimpleNamespace(dual=-2.5))) == 2.5


@pytest.mark.parametrize("solver", ["glpk", "glpk_exact", "gurobi"])
def test_unusable_duals_fall_back_to_the_finite_difference(solver, monkeypatch):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    expected = _finite_difference_coeffs(model, ["E"], 2.0, 0.0)
    module = sys.modules["geckopy.limit_proteins.get_conc_control_coeffs"]
    monkeypatch.setattr(module, "_forward_reduced_cost", lambda rxn: None)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz.tolist() == expected[0].tolist()
    assert coeffs.tolist() == expected[1].tolist()
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


def test_solver_without_duals_uses_the_finite_difference():
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), "scipy")
    _, coeffs = get_conc_control_coeffs(model, fold_change=2.0)
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)
