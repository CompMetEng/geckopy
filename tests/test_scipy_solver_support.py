"""The scipy (HiGHS) solver gives the results of glpk and gurobi.

optlang's scipy interface renumbers its variable columns wrongly after a variable is removed,
which makes cobra read fluxes from the wrong columns, and it raises for duals. geckopy repairs
the first and reads the HiGHS marginals for the second (see ``geckopy.solver_support``).
"""
from pathlib import Path

import cobra
import numpy as np
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import apply_kcat_constraints
from geckopy.limit_proteins import constrain_enz_concs, relax_proteomics_greedy
from geckopy.solver_support import constraint_shadow_prices, repair_scipy_problem, switch_to_scipy
from geckopy.utilities import ec_fva, get_enzyme_bottlenecks

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"
REFERENCE_SOLVERS = ["glpk", "gurobi"]


@pytest.fixture
def scipy_default():
    """cobra builds every new model on scipy while the test runs."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = "scipy"
    yield
    config.solver = previous


def _build(solver="scipy", objective="R3"):
    """ecTestGEM as an ecModel built on ``solver``, kcat 10, with ``objective``; also the
    conventional model it was built from."""
    config = cobra.Configuration()
    previous = config.solver
    try:
        config.solver = solver
    except Exception as exc:  # solver not installed or not licensed
        pytest.skip(f"solver {solver} unavailable: {exc}")
    try:
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        conv = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        model = make_ec_model(conv, adapter)
    finally:
        config.solver = previous
    model.ec.kcat[:] = 10.0
    apply_kcat_constraints(model)
    model.objective = objective
    return model, conv


def _on(solver, tight=False, **kwargs):
    """Like ``_build``, solved on ``solver``. ``tight`` caps every enzyme at 0.01, which makes
    the optimum and its duals unique; without it the LP has several optimal duals and the
    solvers pick different ones."""
    model, conv = _build(solver, **kwargs)
    if tight:
        model.ec.concs[:] = 0.01
        constrain_enz_concs(model)
    try:
        model.solver = solver
        model.optimize()
    except Exception as exc:
        pytest.skip(f"solver {solver} unavailable: {exc}")
    return model, conv


def _columns_are_consistent(model):
    problem = model.solver.problem
    return list(problem._variables.values()) == list(range(len(problem._variables)))


def test_columns_are_consistent_after_make_ec_model(scipy_default):
    model, _ = _build("scipy")
    assert _columns_are_consistent(model)


def test_fluxes_after_make_ec_model_equal_glpk(scipy_default):
    scipy_model, _ = _build("scipy")
    glpk_model, _ = _build("glpk")
    scipy_sol = scipy_model.optimize()
    glpk_sol = glpk_model.optimize()
    assert scipy_sol.objective_value == pytest.approx(glpk_sol.objective_value, rel=1e-7)
    assert scipy_sol.fluxes["R3"] == pytest.approx(scipy_sol.objective_value, rel=1e-7)
    assert scipy_model.reactions.get_by_id("R3").flux == pytest.approx(scipy_sol.objective_value, rel=1e-7)


def test_removing_reactions_keeps_the_fluxes_right(scipy_default):
    model, _ = _build("scipy")
    reference, _ = _build("glpk")
    victims = [r.id for r in model.reactions if r.id.startswith("usage_prot_")][:2]
    for rid in victims:
        model.reactions.get_by_id(rid).delete()
        reference.reactions.get_by_id(rid).delete()
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize(), rel=1e-7)
    assert _columns_are_consistent(model)
    solution = model.optimize()
    assert solution.fluxes["R3"] == pytest.approx(solution.objective_value, rel=1e-7)
    assert model.reactions.get_by_id("R3").flux == pytest.approx(solution.objective_value, rel=1e-7)


def _ring(solver):
    """A plain cobra model on ``solver`` with more variables than constraints."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = solver
    try:
        model = cobra.Model("plain")
        mets = [cobra.Metabolite(f"m{i}") for i in range(3)]
        rxns = []
        for i in range(6):
            rxn = cobra.Reaction(f"r{i}")
            rxn.bounds = (0, 10)
            rxn.add_metabolites({mets[i % 3]: 1.0, mets[(i + 1) % 3]: -1.0} if i else {mets[0]: 1.0})
            rxns.append(rxn)
        model.add_reactions(rxns)
    finally:
        config.solver = previous
    model.objective = "r5"
    return model


def test_a_plain_cobra_model_is_repaired_on_request():
    model, reference = _ring("scipy"), _ring("glpk")
    for m in (model, reference):
        m.remove_reactions(["r1", "r2"])
        m.reactions.get_by_id("r4").bounds = (0, 3)
    assert repair_scipy_problem(model)
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    assert model.reactions.get_by_id("r5").flux == pytest.approx(reference.reactions.get_by_id("r5").flux)


def test_repair_does_nothing_on_other_solvers():
    model = cobra.Model("m")
    model.solver = "glpk"
    assert repair_scipy_problem(model) is False


@pytest.mark.parametrize("reference", REFERENCE_SOLVERS)
def test_ec_fva_on_scipy_equals_the_reference(reference, scipy_default):
    ref_model, ref_conv = _on(reference)
    scipy_model, scipy_conv = _on("scipy")
    expected = ec_fva(ref_model, ref_conv, progress=False, n_proc=1)
    actual = ec_fva(scipy_model, scipy_conv, progress=False, n_proc=1)
    assert list(actual.index) == list(expected.index)
    assert actual.to_numpy(dtype=float) == pytest.approx(expected.to_numpy(dtype=float), abs=1e-6)
    assert np.abs(actual.to_numpy(dtype=float)).max() > 0


@pytest.mark.parametrize("reference", REFERENCE_SOLVERS)
def test_get_enzyme_bottlenecks_on_scipy_equals_the_reference(reference, scipy_default):
    expected = get_enzyme_bottlenecks(_on(reference, tight=True)[0], top=999)
    scipy_model = _on("scipy", tight=True)[0]
    actual = get_enzyme_bottlenecks(scipy_model, top=999)
    assert (expected["shadow_price"].abs() > 1e-9).any()
    assert list(actual.index[:1]) == list(expected.index[:1])
    assert set(actual.index) == set(expected.index)
    aligned = actual.loc[expected.index]
    assert aligned["shadow_price"].to_numpy() == pytest.approx(
        expected["shadow_price"].to_numpy(), abs=1e-6
    )
    assert actual["shadow_price"].abs().is_monotonic_decreasing
    for uniprot, row in actual.iterrows():
        assert row["flux"] == pytest.approx(scipy_model.enzymes.get_by_id(uniprot).flux)


@pytest.mark.parametrize("reference", REFERENCE_SOLVERS)
def test_enzyme_shadow_price_on_scipy_equals_the_reference(reference, scipy_default):
    ref_model, _ = _on(reference, tight=True)
    scipy_model, _ = _on("scipy", tight=True)
    for enz in ref_model.enzymes:
        assert scipy_model.enzymes.get_by_id(enz.id).shadow_price == pytest.approx(
            enz.shadow_price, abs=1e-6
        )


def _tight(model):
    model.ec.concs[:] = 1e-6
    constrain_enz_concs(model)
    return model


@pytest.mark.parametrize("reference", REFERENCE_SOLVERS)
def test_relax_proteomics_greedy_on_scipy_equals_the_reference(reference, scipy_default):
    ref_model, _ = _on(reference)
    scipy_model, _ = _on("scipy")
    expected = relax_proteomics_greedy(_tight(ref_model), minimal_growth=0.001, max_iterations=20)
    actual = relax_proteomics_greedy(_tight(scipy_model), minimal_growth=0.001, max_iterations=20)
    assert expected.converged and actual.converged
    assert [s.relaxed_uniprot for s in actual.trace] == [s.relaxed_uniprot for s in expected.trace]
    assert [s.shadow_price for s in actual.trace] == pytest.approx(
        [s.shadow_price for s in expected.trace], abs=1e-6
    )
    assert actual.final_growth == pytest.approx(expected.final_growth, rel=1e-7)


def test_a_stored_result_from_before_a_change_is_not_read(scipy_default):
    model, _ = _on("scipy", tight=True)
    ids = [enz.prot_metabolite_id for enz in model.enzymes]
    before = constraint_shadow_prices(model, ids)
    usage = next(r for r in model.reactions if r.id.startswith("usage_prot_"))
    usage.upper_bound = usage.upper_bound * 10 + 1.0
    after = constraint_shadow_prices(model, ids)
    reference, _ = _on("glpk", tight=True)
    reference.reactions.get_by_id(usage.id).upper_bound = usage.upper_bound
    reference.optimize()
    expected = np.array([float(reference.constraints[i].dual) for i in ids])
    assert after == pytest.approx(expected, abs=1e-6)
    assert before.shape == after.shape


def _columns_ring(solver):
    """Three metabolites, twelve reactions: removals leave late variables with stale columns."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = solver
    try:
        model = cobra.Model("ring")
        mets = [cobra.Metabolite(f"m{i}") for i in range(3)]
        rxns = []
        for i in range(12):
            rxn = cobra.Reaction(f"r{i}")
            rxn.bounds = (0, 10 + i)
            if i == 0:
                rxn.add_metabolites({mets[0]: 1.0})
            else:
                rxn.add_metabolites({mets[i % 3]: -1.0, mets[(i + 1) % 3]: 0.5 + 0.1 * i})
            rxns.append(rxn)
        model.add_reactions(rxns)
    finally:
        config.solver = previous
    model.objective = "r11"
    return model


def test_removals_after_the_install_keep_every_column_right():
    model, reference = _columns_ring("scipy"), _columns_ring("glpk")
    assert repair_scipy_problem(model)
    for rid in ("r3", "r7", "r9", "r10"):
        model.reactions.get_by_id(rid).delete()
        reference.reactions.get_by_id(rid).delete()
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    solution = model.optimize()
    expected = reference.optimize()
    assert solution.objective_value == pytest.approx(expected.objective_value)
    for rid in solution.fluxes.index:
        assert model.reactions.get_by_id(rid).flux == pytest.approx(solution.fluxes[rid])


def test_a_map_that_is_already_stale_is_rebuilt_from_the_expressions():
    model, reference = _columns_ring("scipy"), _columns_ring("glpk")
    model.reactions.get_by_id("r5").delete()
    reference.reactions.get_by_id("r5").delete()
    model.solver.update()
    assert not _columns_are_consistent(model)
    assert repair_scipy_problem(model)
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())


def test_the_sparse_problem_hands_highs_the_matrix_of_the_dense_problem():
    model = _columns_ring("scipy")
    dense = model.solver.problem.A.copy()
    assert repair_scipy_problem(model)
    assert (model.solver.problem.A.toarray() == dense).all()


def test_switch_to_scipy_keeps_the_objective_and_its_direction():
    model, reference = _columns_ring("glpk"), _columns_ring("glpk")
    for m in (model, reference):
        m.objective = "r8"
        m.objective_direction = "min"
        m.reactions.get_by_id("r1").lower_bound = 2.0
    switch_to_scipy(model)
    assert model.solver.interface.__name__.endswith("scipy_interface")
    assert model.objective_direction == "min"
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    assert model.reactions.get_by_id("r8").flux == pytest.approx(model.slim_optimize())


def test_load_conventional_gem_and_make_ec_model_end_on_scipy(scipy_default):
    from geckopy import load_conventional_gem

    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    conv = load_conventional_gem(adapter)
    assert conv.solver.interface.__name__.endswith("scipy_interface")
    reference = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    for m in (conv, reference):
        m.reactions.get_by_id("S1").bounds = (-10.0, 10.0)
        m.objective = "R5"
    assert conv.slim_optimize() == pytest.approx(reference.slim_optimize())
    assert reference.slim_optimize() > 0
    model = make_ec_model(conv, adapter)
    assert model.solver.interface.__name__.endswith("scipy_interface")
    assert cobra.Configuration().solver.__name__.endswith("scipy_interface")


def test_shadow_prices_need_an_optimal_solution(scipy_default):
    model, _ = _on("scipy", tight=True)
    model.reactions.get_by_id("R3").lower_bound = 1e6
    with pytest.raises(RuntimeError, match="optimal"):
        constraint_shadow_prices(model, [model.enzymes.get_by_id(model.ec.enzymes[0]).prot_metabolite_id])


@pytest.mark.parametrize("status_case", ["infeasible", "unbounded"])
def test_a_solve_without_optimum_returns_a_solution_with_its_status(status_case):
    config = cobra.Configuration()
    previous = config.solver
    config.solver = "scipy"
    try:
        model = cobra.Model("chain")
        met = cobra.Metabolite("m")
        source, sink = cobra.Reaction("source"), cobra.Reaction("sink")
        source.add_metabolites({met: 1.0})
        sink.add_metabolites({met: -1.0})
        source.bounds = (0, 10) if status_case == "infeasible" else (0, 1000.0)
        sink.bounds = (20, 30) if status_case == "infeasible" else (0, 1000.0)
        model.add_reactions([source, sink])
    finally:
        config.solver = previous
    model.objective = "sink"
    if status_case == "unbounded":
        source.upper_bound = float("inf")
        sink.upper_bound = float("inf")
    assert repair_scipy_problem(model)
    if status_case == "unbounded":
        # cobra raises for this status on every solver
        with pytest.raises(cobra.exceptions.OptimizationError, match="unbounded"):
            model.optimize(raise_error=False)
        return
    solution = model.optimize(raise_error=False)
    assert solution.status == status_case
    assert np.isnan(solution.objective_value)
    assert solution.fluxes.isna().all()


def test_setting_the_solver_of_an_ec_model_to_scipy_gives_the_sparse_problem():
    from geckopy.solver_support import _SparseProblem

    model, _ = _build("glpk")
    expected = model.slim_optimize()
    model.solver = "scipy"
    assert isinstance(model._solver.problem, _SparseProblem)
    assert model.slim_optimize() == pytest.approx(expected)


def test_load_ec_model_ends_on_the_sparse_scipy_problem(scipy_default, tmp_path):
    from geckopy import load_ec_model, save_ec_model
    from geckopy.solver_support import _SparseProblem

    model, _ = _build("glpk")
    expected = model.slim_optimize()
    save_ec_model(model, tmp_path / "ecModel.yml")
    loaded = load_ec_model(tmp_path / "ecModel.yml")
    assert isinstance(loaded._solver.problem, _SparseProblem)
    assert cobra.Configuration().solver.__name__.endswith("scipy_interface")
    loaded.objective = "R3"
    assert loaded.slim_optimize() == pytest.approx(expected)


@pytest.mark.parametrize("how", ["pickle", "deepcopy", "copy"])
def test_a_scipy_ec_model_pickles_and_copies_into_the_sparse_problem(how):
    import copy
    import pickle

    from geckopy.solver_support import _SparseProblem

    model, _ = _build("glpk")
    expected = model.slim_optimize()
    model.solver = "scipy"
    model.slim_optimize()
    clone = {
        "pickle": lambda: pickle.loads(pickle.dumps(model)),
        "deepcopy": lambda: copy.deepcopy(model),
        "copy": model.copy,
    }[how]()
    assert isinstance(clone._solver.problem, _SparseProblem)
    assert clone.solver.interface.__name__.endswith("scipy_interface")
    assert clone.slim_optimize() == pytest.approx(expected)
    assert clone.reactions.get_by_id("R3").flux == pytest.approx(model.reactions.get_by_id("R3").flux)
