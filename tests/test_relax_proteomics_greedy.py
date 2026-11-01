"""Tests for relax_proteomics_greedy."""
from pathlib import Path

import cobra
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import apply_kcat_constraints
from geckopy.limit_proteins import (
    GreedyRelaxResult,
    constrain_enz_concs,
    relax_proteomics_greedy,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


def _ec_with_tight_concs(growth_target: float = 0.5):
    """ecTestGEM with all concs set to a tight value that under-supplies
    a growth target. Returns (model, target)."""
    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    model = make_ec_model(cobra_model, adapter)
    model.ec.kcat[:] = 10.0
    apply_kcat_constraints(model)
    model.objective = "R3"
    # Tighten every enzyme to a low value to force infeasibility at target.
    for i in range(len(model.ec.enzymes)):
        model.ec.concs[i] = 1e-6
    constrain_enz_concs(model)
    return model, growth_target


def test_converges_on_solvable_case():
    model, target = _ec_with_tight_concs(growth_target=0.001)
    result = relax_proteomics_greedy(
        model, minimal_growth=target, max_iterations=20,
    )
    assert isinstance(result, GreedyRelaxResult)
    assert result.converged
    assert result.final_growth >= target - 1e-9


def test_relaxed_dict_records_originals():
    model, _ = _ec_with_tight_concs()
    result = relax_proteomics_greedy(
        model, minimal_growth=0.001, max_iterations=20,
    )
    for uniprot, original in result.relaxed.items():
        # All originals were 1e-6 from the fixture.
        assert original == pytest.approx(1e-6)


def test_trace_records_each_iteration():
    model, _ = _ec_with_tight_concs()
    result = relax_proteomics_greedy(
        model, minimal_growth=0.001, max_iterations=20,
    )
    assert len(result.trace) == len(result.relaxed)
    for i, step in enumerate(result.trace):
        assert step.iteration == i
        assert step.relaxed_uniprot in result.relaxed


def test_no_candidates_returns_not_converged():
    """When no enzymes are proteomics-constrained, there are no
    candidates; the function returns immediately as not-converged
    (unless the model already reaches the target)."""
    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    model = make_ec_model(cobra_model, adapter)
    model.ec.kcat[:] = 10.0
    apply_kcat_constraints(model)
    model.objective = "R3"
    # No concs set => no candidates eligible.
    target = 1e9  # unreachable, forces the not-converged path
    result = relax_proteomics_greedy(model, minimal_growth=target)
    assert result.converged is False
    assert result.trace == []
    assert result.relaxed == {}


def test_enzyme_set_filter():
    """With enzyme_set, only listed enzymes are eligible to relax."""
    model, _ = _ec_with_tight_concs()
    only = {"P4"}
    result = relax_proteomics_greedy(
        model, minimal_growth=0.001,
        enzyme_set=only, max_iterations=20,
    )
    assert set(result.relaxed.keys()).issubset(only)


def test_max_iterations_raises():
    """Setting max_iterations=1 on a model needing more raises."""
    model, _ = _ec_with_tight_concs()
    # Set the target high enough that one relaxation cannot satisfy it.
    with pytest.raises(RuntimeError, match="Did not converge"):
        relax_proteomics_greedy(
            model, minimal_growth=1e9, max_iterations=1,
        )


def test_empty_enzyme_set_makes_no_enzyme_eligible():
    model, _ = _ec_with_tight_concs()
    result = relax_proteomics_greedy(
        model, minimal_growth=0.001, enzyme_set=[], max_iterations=20,
    )
    assert result.relaxed == {}
    assert result.trace == []
