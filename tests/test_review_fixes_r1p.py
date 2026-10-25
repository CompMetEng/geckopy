"""Edge cases of get_conc_control_coeffs, fit_sigma and the docstring of fill_kcats_from_isozymes."""
import numpy as np
import pytest

from geckopy.kcat_tuning import fit_sigma
from geckopy.limit_proteins import get_conc_control_coeffs
from geckopy.limit_proteins.get_conc_control_coeffs import _finite_difference_coeffs

from test_get_conc_control_coeffs import _build_toy_ec_model
from test_sigma_fitter import _adapter, _build_pool_only_model

SOLVERS = ["glpk", "glpk_exact", "gurobi"]


def _with_solver(model, solver):
    try:
        model.solver = solver
        model.optimize()
    except Exception as exc:  # solver not installed or not licensed
        pytest.skip(f"solver {solver} unavailable: {exc}")
    return model


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_small_coefficient_on_a_small_bound_is_kept(solver):
    # One unit of enzyme per 2000 units of flux: growth = ub / 2000, so the coefficient is 5e-4
    # while a 0.01 percent rise of a bound of 1e-3 raises growth by only 5e-11.
    model = _with_solver(_build_toy_ec_model(enzyme_ub=1e-3, enzyme_coeff_in_R=-2000.0), solver)
    assert model.optimize().objective_value == pytest.approx(5e-7, rel=1e-6)
    enz, coeffs = get_conc_control_coeffs(model)
    _, local = _finite_difference_coeffs(model, ["E"], 2.0, 0.0)
    assert enz[0]
    assert local[0] == pytest.approx(5e-4, rel=1e-6)
    assert coeffs[0] == pytest.approx(5e-4, rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_large_coefficient_is_unchanged(solver):
    model = _with_solver(_build_toy_ec_model(enzyme_ub=5.0), solver)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("method", ["grid", "bisect"])
def test_a_model_that_cannot_grow_gives_the_first_grid_sigma(tmp_path, method):
    adapter = _adapter(tmp_path, p_tot=0.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, growth_rate=0.5, p_tot=0.0, f=1.0, method=method)
    assert result.sigma == pytest.approx(0.01)
    assert model.reactions.get_by_id("prot_pool_exchange").upper_bound == 0.0
