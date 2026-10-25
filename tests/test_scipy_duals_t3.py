"""get_conc_control_coeffs with optlang's scipy backend: the reduced cost comes from HiGHS's bound marginal.

optlang raises on ``Variable.dual`` for the scipy solver but keeps the ``linprog`` result, whose
``upper.marginals`` hold d(objective)/d(upper bound) of the minimised problem. The function reads
that value, so scipy takes the same one-solve path as the other solvers. A result without usable
marginals, or one that does not belong to the current variables, falls back to the
finite-difference loop.
"""
import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from geckopy.limit_proteins import get_conc_control_coeffs
from geckopy.limit_proteins.get_conc_control_coeffs import (
    _finite_difference_coeffs,
    _forward_reduced_cost,
    _scipy_upper_marginal,
)

from test_conc_control_coeffs_shadow import _degenerate_model, _pool_binding_model, _with_solver
from test_get_conc_control_coeffs import _build_toy_ec_model

mod = importlib.import_module("geckopy.limit_proteins.get_conc_control_coeffs")


def _forbid_finite_difference(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the finite-difference loop was used")

    monkeypatch.setattr(mod, "_finite_difference_coeffs", fail)


def _scipy_model(**kwargs):
    return _with_solver(_build_toy_ec_model(**kwargs), "scipy")


def _flux(model, rxn):
    """The flux cobra reports for ``rxn`` in a fresh solve of the model."""
    return float(model.optimize().fluxes[rxn.id])


def test_scipy_reads_the_bound_marginal_and_not_the_finite_difference(monkeypatch):
    model = _scipy_model(enzyme_ub=5.0)
    _forbid_finite_difference(monkeypatch)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


def test_scipy_marginal_of_a_binding_bound_equals_the_marginal_growth():
    model = _scipy_model(enzyme_ub=5.0, enzyme_coeff_in_R=-2.0)
    rxn = model.reactions.get_by_id("usage_prot_E")
    assert _forward_reduced_cost(rxn, _flux(model, rxn)) == pytest.approx(0.5, rel=1e-6)
    _, local = _finite_difference_coeffs(model, ["E"], 1.001, 0.0)
    assert local[0] == pytest.approx(0.5, rel=1e-6)


def test_scipy_gives_zero_for_a_bound_that_does_not_bind_while_the_pool_binds(monkeypatch):
    model = _pool_binding_model("scipy")
    rxn = model.reactions.get_by_id("usage_prot_E")
    _forbid_finite_difference(monkeypatch)
    assert _forward_reduced_cost(rxn, _flux(model, rxn)) == 0.0
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == 0.0


def test_scipy_degenerate_vertex_is_resolved_by_the_verification_step(monkeypatch):
    model = _degenerate_model("scipy")
    _forbid_finite_difference(monkeypatch)
    _, coeffs = get_conc_control_coeffs(model)
    assert coeffs[0] == 0.0


@pytest.mark.parametrize("other", ["glpk", "gurobi"])
@pytest.mark.parametrize(
    "build",
    [
        lambda s: _with_solver(_build_toy_ec_model(enzyme_ub=5.0), s),
        lambda s: _with_solver(_build_toy_ec_model(enzyme_ub=5.0, enzyme_coeff_in_R=-2.0), s),
        _pool_binding_model,
        _degenerate_model,
    ],
    ids=["binding", "coefficient_two", "pool_binding", "degenerate"],
)
def test_scipy_agrees_with_the_other_solvers(build, other):
    reference = build(other)
    scipy_model = build("scipy")
    enz_ref, ref = get_conc_control_coeffs(reference)
    enz_sp, sp = get_conc_control_coeffs(scipy_model)
    np.testing.assert_array_equal(enz_ref, enz_sp)
    np.testing.assert_allclose(sp, ref, rtol=1e-6, atol=1e-12)


def test_scipy_result_without_bound_marginals_falls_back(caplog, monkeypatch):
    model = _scipy_model(enzyme_ub=8.0)
    model.reactions.get_by_id("EX_A").lower_bound = -10.0
    problem_class = type(model.solver.problem)
    original = problem_class.optimize

    def optimize_without_marginals(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._solution = SimpleNamespace(x=self._solution.x, status=self._solution.status)

    monkeypatch.setattr(problem_class, "optimize", optimize_without_marginals)
    mod._FALLBACK_REPORTED.clear()
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        pm, pc = get_conc_control_coeffs(model, fold_change=2.0)
    fm, fc = _finite_difference_coeffs(model, ["E"], 2.0, 0.0)
    np.testing.assert_array_equal(pm, fm)
    np.testing.assert_allclose(pc, fc, rtol=1e-6, atol=1e-12)
    assert "returns no usable variable duals" in caplog.text


def test_scipy_result_that_does_not_belong_to_the_current_variables_gives_none():
    model = _scipy_model(enzyme_ub=5.0)
    rxn = model.reactions.get_by_id("usage_prot_E")
    flux = _flux(model, rxn)
    problem = model.solver.problem
    assert _scipy_upper_marginal(rxn, flux) is not None
    assert _scipy_upper_marginal(rxn, flux + 1.0) is None  # not the solution cobra returned
    assert _scipy_upper_marginal(rxn, None) is None
    problem._solution.x[:] = 0.0  # a stored result that is not the current solution
    assert _scipy_upper_marginal(rxn, flux) is None


def test_nan_marginal_falls_back(monkeypatch):
    model = _scipy_model(enzyme_ub=5.0)
    monkeypatch.setattr(mod, "_scipy_upper_marginal", lambda rxn, flux: float("nan"))
    rxn = model.reactions.get_by_id("usage_prot_E")
    assert _forward_reduced_cost(rxn, 5.0) is None
    enz, coeffs = get_conc_control_coeffs(model)
    _, local = _finite_difference_coeffs(model, ["E"], 2.0, 0.0)
    assert coeffs[0] == pytest.approx(local[0], rel=1e-6)
