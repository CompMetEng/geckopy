"""Tests for get_enzyme_bottlenecks."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import ModelAdapter, make_ec_model
from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.ec_model.pipeline import apply_kcat_constraints
from geckopy.utilities import get_enzyme_bottlenecks

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


_ECTESTGEM_CACHE = None


def _ectestgem_solvable():
    """Cached build of the ecTestGEM ecModel with kcat=10 + R3 objective;
    deep-copied per call so per-test mutations are isolated."""
    import copy as _copy
    global _ECTESTGEM_CACHE
    if _ECTESTGEM_CACHE is None:
        from geckopy import EcModel  # local import keeps the type annotation light
        adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
        cobra_model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
        ec_model: EcModel = make_ec_model(cobra_model, adapter)
        ec_model.ec.kcat[:] = 10.0
        apply_kcat_constraints(ec_model)
        ec_model.objective = "R3"
        _ECTESTGEM_CACHE = ec_model
    return _copy.deepcopy(_ECTESTGEM_CACHE)


def test_returns_top_n():
    model = _ectestgem_solvable()
    df = get_enzyme_bottlenecks(model, top=3)
    assert len(df) == 3


def test_returns_at_most_top_when_fewer_enzymes():
    model = _ectestgem_solvable()
    df = get_enzyme_bottlenecks(model, top=999)
    assert len(df) == len(model.ec.enzymes)


def test_sorted_by_objective_gain():
    model = _ectestgem_solvable()
    df = get_enzyme_bottlenecks(model, top=999)
    rounded = df["objective_gain"].round(9).to_list()
    assert rounded == sorted(rounded, reverse=True)
    assert (df["objective_gain"] >= 0).all()


def test_columns():
    model = _ectestgem_solvable()
    df = get_enzyme_bottlenecks(model, top=3)
    assert list(df.columns) == [
        "gene", "objective_gain", "shadow_price", "flux", "cap_usage",
        "upper_bound", "multiple_active",
    ]
    assert df.index.name == "uniprot"
    assert "n_resolved" in df.attrs
    assert "active_constraints" in df.attrs


def _toy(solver="glpk", *, ubs=(4.0, 1.0, 5.0, 3.0), pool_ub=1e6):
    """Four enzymes. E1 and E2 are parallel steps A -> B, E3 turns B into C, E4 turns A into D.
    The objective is SK_C + SK_D, the substrate is not limiting and R4 is capped at 1000.

    With the default bounds E1, E2 and E3 are at their bound (4 + 1 = 5) and raising any one of
    them alone leaves the optimum unchanged, while E4 is at its bound and raising it adds flux.
    """
    model = EcModel("toy")
    mets = {n: cobra.Metabolite(n, compartment="c") for n in
            ["A_e", "A", "B", "C", "D", "pool", "prot_E1", "prot_E2", "prot_E3", "prot_E4"]}
    model.add_metabolites(list(mets.values()))

    def rxn(rid, stoich, lb, ub):
        r = cobra.Reaction(rid)
        r.add_metabolites({mets[k]: v for k, v in stoich.items()})
        r.bounds = (lb, ub)
        return r

    reactions = [
        rxn("EX_A", {"A_e": -1.0}, -1e6, 0.0),
        rxn("TR_A", {"A_e": -1.0, "A": 1.0}, 0.0, 1e6),
        rxn("R1", {"A": -1.0, "B": 1.0, "prot_E1": -1.0}, 0.0, 1000.0),
        rxn("R2", {"A": -1.0, "B": 1.0, "prot_E2": -1.0}, 0.0, 1000.0),
        rxn("R3", {"B": -1.0, "C": 1.0, "prot_E3": -1.0}, 0.0, 1000.0),
        rxn("R4", {"A": -1.0, "D": 1.0, "prot_E4": -1.0}, 0.0, 1000.0),
        rxn("SK_C", {"C": -1.0}, 0.0, 1000.0),
        rxn("SK_D", {"D": -1.0}, 0.0, 1000.0),
        rxn("prot_pool_exchange", {"pool": 1.0}, 0.0, pool_ub),
    ]
    for i, ub in enumerate(ubs, start=1):
        reactions.append(rxn(f"usage_prot_E{i}", {"pool": -1.0, f"prot_E{i}": 1.0}, 0.0, ub))
    model.add_reactions(reactions)
    model.objective = {model.reactions.SK_C: 1.0, model.reactions.SK_D: 1.0}
    model.ec = EcData(
        rxns=["R1", "R2", "R3", "R4"],
        kcat=np.ones(4),
        source=[""] * 4,
        notes=[""] * 4,
        eccodes=[""] * 4,
        genes=["g1", "g2", "g3", "g4"],
        enzymes=["E1", "E2", "E3", "E4"],
        mw=np.full(4, 100.0),
        sequence=[""] * 4,
        concs=np.array(ubs, dtype=float),
        rxn_enz_mat=sparse.identity(4, format="csr"),
    )
    try:
        model.solver = solver
    except Exception as exc:
        pytest.skip(f"solver {solver} unavailable: {exc}")
    return model


def test_toy_ranks_by_resolved_gain_not_by_dual():
    model = _toy()
    df = get_enzyme_bottlenecks(model, top=10)
    assert list(df.index) == ["E4", "E1", "E2", "E3"]
    assert df.loc["E4", "objective_gain"] == pytest.approx(3.0)
    assert df.loc[["E1", "E2", "E3"], "objective_gain"].to_list() == [0.0, 0.0, 0.0]
    assert df.attrs["n_resolved"] == 4
    assert df.attrs["active_constraints"] == 4
    assert df["multiple_active"].all()
    assert ((df["cap_usage"] - 1.0).abs() < 1e-6).all()
    # A zero gain with a nonzero dual: the dual is reported but does not enter the order.
    assert (df.loc[["E1", "E2", "E3"], "shadow_price"].abs() > 1e-9).any()
    # The model is restored.
    assert model.reactions.usage_prot_E4.upper_bound == 3.0
    assert model.enzymes.get_by_id("E4").flux == pytest.approx(3.0)


def test_non_binding_enzymes_are_not_resolved():
    model = _toy(ubs=(4.0, 3.0, 20.0, 5000.0))
    df = get_enzyme_bottlenecks(model, top=10)
    # E1 and E2 carry 7 through E3 (ub 20) and are at their bound; E3 is not, and E4 is limited by R4.
    assert df.attrs["n_resolved"] == 2
    assert df.attrs["active_constraints"] == 2
    assert df.loc[["E3", "E4"], "objective_gain"].to_list() == [0.0, 0.0]
    assert df.loc["E1", "objective_gain"] > 0
    assert df.loc["E1", "multiple_active"]
    assert list(df.index[:2]) == ["E1", "E2"]


def test_single_active_constraint_is_not_multiple_active():
    model = _toy(ubs=(10.0, 10.0, 2.0, 5000.0))
    df = get_enzyme_bottlenecks(model, top=10)
    assert df.attrs["n_resolved"] == 1
    assert df.attrs["active_constraints"] == 1
    assert df.loc["E3", "objective_gain"] == pytest.approx(2.0)
    assert not df.loc["E3", "multiple_active"]
    assert df.loc[["E1", "E2", "E4"], "multiple_active"].all()
    assert list(df.index)[0] == "E3"


def test_pool_exchange_at_its_bound_is_an_active_constraint():
    model = _toy(ubs=(10.0, 10.0, 2.0, 5000.0), pool_ub=2.0)
    df = get_enzyme_bottlenecks(model, top=10)
    assert df.attrs["active_constraints"] == 1
    assert df.attrs["n_resolved"] == 0
    assert df["multiple_active"].all()
    assert (df["objective_gain"] == 0).all()


def test_pool_exchange_and_an_enzyme_at_their_bounds():
    model = _toy(ubs=(10.0, 10.0, 2.0, 5000.0), pool_ub=1004.0)
    df = get_enzyme_bottlenecks(model, top=10)
    assert df.attrs["active_constraints"] == 2
    assert df.attrs["n_resolved"] == 1
    assert df.loc["E3", "multiple_active"]
    assert df.loc["E3", "objective_gain"] == 0.0


def test_fold_change_is_respected():
    twice = get_enzyme_bottlenecks(_toy(), top=1, fold_change=2.0)
    thrice = get_enzyme_bottlenecks(_toy(), top=1, fold_change=3.0)
    assert twice.index[0] == thrice.index[0] == "E4"
    assert twice["objective_gain"].iloc[0] == pytest.approx(3.0)
    assert thrice["objective_gain"].iloc[0] == pytest.approx(6.0)


@pytest.mark.parametrize("fold_change", [1.0, 0.5, float("nan")])
def test_fold_change_must_exceed_one(fold_change):
    with pytest.raises(ValueError, match="fold_change"):
        get_enzyme_bottlenecks(_toy(), fold_change=fold_change)


def test_gain_is_the_improvement_of_a_minimised_objective():
    maximised = get_enzyme_bottlenecks(_toy(), top=4)
    model = _toy()
    model.objective = {model.reactions.get_by_id("SK_C"): -1.0, model.reactions.get_by_id("SK_D"): -1.0}
    model.objective_direction = "min"
    minimised = get_enzyme_bottlenecks(model, top=4)
    assert list(minimised.index) == list(maximised.index)
    assert minimised["objective_gain"].to_numpy() == pytest.approx(maximised["objective_gain"].to_numpy())


@pytest.mark.parametrize("solver", ["glpk", "scipy", "gurobi"])
def test_order_does_not_depend_on_the_solver(solver):
    reference = get_enzyme_bottlenecks(_toy("glpk"), top=10)
    df = get_enzyme_bottlenecks(_toy(solver), top=10)
    assert list(df.index) == list(reference.index)
    assert df["objective_gain"].to_numpy() == pytest.approx(
        reference["objective_gain"].to_numpy(), abs=1e-6
    )
    assert df["multiple_active"].to_list() == reference["multiple_active"].to_list()
    assert df.attrs["n_resolved"] == reference.attrs["n_resolved"]


def test_raises_on_infeasible():
    model = _ectestgem_solvable()
    # Make infeasible: require biomass higher than achievable.
    r3 = model.reactions.get_by_id("R3")
    r3.lower_bound = 1e9
    with pytest.raises(RuntimeError, match="Solver status"):
        get_enzyme_bottlenecks(model, top=3)


def test_warns_when_no_enzyme_is_at_its_bound(caplog):
    model = _toy(ubs=(1e5, 1e5, 1e5, 1e5))
    with caplog.at_level("WARNING", logger="geckopy.utilities.bottlenecks"):
        df = get_enzyme_bottlenecks(model)
    assert df.attrs["n_resolved"] == 0
    assert (df["objective_gain"] == 0).all()
    assert "no enzyme is at its usage bound" in caplog.text
    assert "no capacity constraint is active" in caplog.text


def test_warning_names_the_protein_pool_when_it_is_the_active_constraint(caplog):
    model = _toy(ubs=(1e5, 1e5, 1e5, 1e5), pool_ub=6.0)
    with caplog.at_level("WARNING", logger="geckopy.utilities.bottlenecks"):
        df = get_enzyme_bottlenecks(model)
    assert df.attrs["n_resolved"] == 0
    assert df.attrs["active_constraints"] == 1
    assert "the protein pool exchange is at its bound" in caplog.text


def test_no_warning_when_an_enzyme_is_at_its_bound(caplog):
    with caplog.at_level("WARNING", logger="geckopy.utilities.bottlenecks"):
        get_enzyme_bottlenecks(_toy())
    assert "no enzyme is at its usage bound" not in caplog.text
