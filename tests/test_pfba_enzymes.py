"""Tests for pfba_enzymes."""
from pathlib import Path

import cobra
import pytest

from geckopy import EcModel, ModelAdapter, make_ec_model
from geckopy.ec_model.constants import USAGE_PREFIX
from geckopy.ec_model.pipeline import apply_kcat_constraints
from geckopy.utilities import pfba_enzymes

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


_ECTESTGEM_CACHE: EcModel | None = None


def _ectestgem_ec_model_with_kcats() -> EcModel:
    """ecTestGEM with realistic kcats so usage flows. Cached at module
    scope; deep-copied per call."""
    import copy as _copy
    global _ECTESTGEM_CACHE
    if _ECTESTGEM_CACHE is None:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        ec_model = make_ec_model(cobra_model, adapter)
        # All kcats default to 10 /s so usage coefficients are non-zero.
        ec_model.ec.kcat[:] = 10.0
        apply_kcat_constraints(ec_model)
        ec_model.objective = "R3"  # maximise R3
        _ECTESTGEM_CACHE = ec_model
    return _copy.deepcopy(_ECTESTGEM_CACHE)


def _total_usage(sol: cobra.Solution) -> float:
    return sum(
        abs(v) for k, v in sol.fluxes.items() if k.startswith(USAGE_PREFIX)
    )


def test_returns_solution():
    model = _ectestgem_ec_model_with_kcats()
    sol = pfba_enzymes(model)
    assert isinstance(sol, cobra.Solution)
    assert sol.status == "optimal"


def test_objective_preserved_at_full_fraction():
    model = _ectestgem_ec_model_with_kcats()
    fba_optimum = model.slim_optimize()
    sol = pfba_enzymes(model, fraction_of_optimum=1.0)
    assert sol.fluxes["R3"] == pytest.approx(fba_optimum, abs=1e-6)


def test_default_fixes_the_objective_at_0_999999_of_the_optimum():
    """MATLAB's getPfbaEnzymes (RAVEN's solveLP) keeps 0.999999 of the optimum."""
    model = _ectestgem_ec_model_with_kcats()
    fba_optimum = model.slim_optimize()
    sol = pfba_enzymes(model)
    ratio = sol.fluxes["R3"] / fba_optimum
    assert ratio == pytest.approx(0.999999, abs=2e-7)


def test_objective_preserved_at_partial_fraction():
    model = _ectestgem_ec_model_with_kcats()
    fba_optimum = model.slim_optimize()
    sol = pfba_enzymes(model, fraction_of_optimum=0.5)
    assert sol.fluxes["R3"] >= 0.5 * fba_optimum - 1e-6


def test_total_usage_le_plain_fba():
    """pfba_enzymes total usage should be at most the plain FBA total."""
    model = _ectestgem_ec_model_with_kcats()
    fba_sol = model.optimize()
    pfba_sol = pfba_enzymes(model)
    assert _total_usage(pfba_sol) <= _total_usage(fba_sol) + 1e-9


def test_total_usage_le_plain_pfba():
    """pfba_enzymes should beat cobra.pfba on enzyme usage by construction."""
    model = _ectestgem_ec_model_with_kcats()
    cobra_pfba_sol = cobra.flux_analysis.pfba(model)
    pfba_enz_sol = pfba_enzymes(model)
    assert _total_usage(pfba_enz_sol) <= _total_usage(cobra_pfba_sol) + 1e-9


def test_raises_on_double_call():
    """Nested pfba_enzymes(pfba_enzymes(...)) raises ValueError because the
    objective name from a prior call would still be set. We mimic this by
    setting the objective name to the sentinel."""
    model = _ectestgem_ec_model_with_kcats()
    # Build an objective with the sentinel name.
    biomass = model.reactions.get_by_id("R3")
    new_obj = model.problem.Objective(
        biomass.flux_expression, direction="max",
        name="_pfba_enzymes_objective",
    )
    model.objective = new_obj
    with pytest.raises(ValueError, match="_pfba_enzymes_objective"):
        pfba_enzymes(model)


def test_gecko_light_raises():
    model = _ectestgem_ec_model_with_kcats()
    model.ec.gecko_light = True
    with pytest.raises(NotImplementedError):
        pfba_enzymes(model)
