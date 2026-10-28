"""Tests for sensitivity_tuning."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.kcat_tuning import (
    TunedKcatsResult,
    sensitivity_tuning,
)


# --------------------------------------------------------------------------- #
# Tiny enzyme-constrained model fixture
# --------------------------------------------------------------------------- #

def _adapter(
    tmp_path: Path, *, gr_exp: float = 0.5, bio_rxn: str = "biomass",
) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\n'
        f'org_name = "test"\n'
        f'gr_exp = {gr_exp}\n'
        f'bio_rxn = "{bio_rxn}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_toy(
    adapter: ModelAdapter,
    *,
    initial_kcat: float = 1.0,
    enzyme_mw: float = 100.0,
) -> EcModel:
    """One-step model: A -> biomass, gated by enzyme E.

    Stoichiometry of R: A_c + (mw / (kcat * 3600)) * prot_E -> biomass.
    With pool size = 1 mg/gDW, growth = 1 / (mw / (kcat * 3600))
    = (kcat * 3600) / mw.

    With kcat=1, mw=100: growth = 36/gDW/h.
    Bumping kcat by 10x: growth becomes 360 (now bounded by upstream).
    """
    model = EcModel("toy", adapter=adapter)

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    model.add_metabolites([A_e, A_c, pool, prot_E, bio_met])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -1000.0
    EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0
    TR_A.upper_bound = 1000.0

    R = cobra.Reaction("R")
    coeff = enzyme_mw / (initial_kcat * 3600.0)  # mg/mmol per unit flux
    R.add_metabolites({A_c: -1.0, prot_E: -coeff, bio_met: 1.0})
    R.lower_bound = 0.0
    R.upper_bound = 1000.0

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.lower_bound = 0.0
    BIO.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0
    pool_ex.upper_bound = 1.0  # 1 mg/gDW pool

    usage_E = cobra.Reaction("usage_prot_E")
    usage_E.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage_E.lower_bound = 0.0
    usage_E.upper_bound = 1000.0

    model.add_reactions([EX_A, TR_A, R, BIO, pool_ex, usage_E])
    model.objective = "biomass"

    model.ec = EcData(
        rxns=["R"],
        kcat=np.array([initial_kcat]),
        source=["initial"],
        notes=[""],
        eccodes=[""],
        genes=["g_E"],
        enzymes=["E"],
        mw=np.array([enzyme_mw]),
        sequence=[""],
        concs=np.array([np.nan]),
        rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_no_adapter_no_growth_rate_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_toy(adapter)
    model.adapter = None
    with pytest.raises(ValueError, match="desired_growth_rate"):
        sensitivity_tuning(model)


# --------------------------------------------------------------------------- #
# Happy path: kcat is bumped to reach growth target
# --------------------------------------------------------------------------- #

def test_kcat_increased_to_reach_growth_target(tmp_path):
    """Initial kcat=1 -> growth=36; target=300 needs ~10x kcat."""
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0, enzyme_mw=100.0)
    initial_kcat = float(model.ec.kcat[0])

    result = sensitivity_tuning(model, fold_change=10.0)

    assert isinstance(result, TunedKcatsResult)
    assert "R" in result.rxns
    # Kcat must be at least 10x to reach growth=300.
    assert model.ec.kcat[0] >= initial_kcat * 10


def test_growth_target_actually_reached(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0, enzyme_mw=100.0)
    sensitivity_tuning(model, fold_change=10.0)
    sol = model.optimize()
    assert sol.objective_value >= 300.0


def test_explicit_growth_rate_overrides_adapter(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=999.0)
    model = _build_toy(adapter)
    sensitivity_tuning(model, desired_growth_rate=300.0, fold_change=10.0)
    sol = model.optimize()
    assert sol.objective_value >= 300.0


# --------------------------------------------------------------------------- #
# Notes annotation
# --------------------------------------------------------------------------- #

def test_notes_annotated_with_pre_tune_kcat(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    sensitivity_tuning(model, fold_change=10.0)
    note = model.ec.notes[0]
    assert "preTuneKcat=" in note
    assert "source:initial" in note


def test_source_set_to_sensitivity_tuning(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter)
    sensitivity_tuning(model, fold_change=10.0)
    assert model.ec.source[0] == "sensitivityTuning"


def test_re_tuning_does_not_double_annotate(tmp_path):
    """A second tuning pass on an already-tuned kcat should not
    re-annotate (source is 'sensitivityTuning')."""
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    sensitivity_tuning(model, fold_change=10.0)
    note_after_first = model.ec.notes[0]

    # Bump target to force more tuning.
    sensitivity_tuning(
        model, desired_growth_rate=3000.0, fold_change=10.0,
    )
    note_after_second = model.ec.notes[0]

    # Note should be unchanged (no second "preTuneKcat=" appended).
    assert note_after_first == note_after_second
    assert note_after_second.count("preTuneKcat=") == 1


# --------------------------------------------------------------------------- #
# prot_to_ignore
# --------------------------------------------------------------------------- #

def test_prot_to_ignore_excludes_enzyme(tmp_path):
    """If the only limiting enzyme is in `prot_to_ignore`, no tuning
    should happen and the function should warn and break."""
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter)
    initial_kcat = float(model.ec.kcat[0])

    result = sensitivity_tuning(model, prot_to_ignore=["E"])

    assert len(result.rxns) == 0
    assert model.ec.kcat[0] == initial_kcat


# --------------------------------------------------------------------------- #
# Already at target
# --------------------------------------------------------------------------- #

def test_already_at_target_no_tuning_needed(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=10.0)  # very low target
    model = _build_toy(adapter, initial_kcat=1.0, enzyme_mw=100.0)
    # Initial growth = 36 > 10 -> no tuning.
    result = sensitivity_tuning(model)
    assert len(result.rxns) == 0


# --------------------------------------------------------------------------- #
# Return value contents
# --------------------------------------------------------------------------- #

def test_result_has_expected_fields_populated(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    result = sensitivity_tuning(model, fold_change=10.0)

    assert len(result.rxns) == 1
    assert result.rxn_names[0] == "R"
    assert result.enzymes[0] == "E"
    assert result.old_kcat[0] == pytest.approx(1.0)
    assert result.new_kcat[0] >= 10.0
    assert result.source[0] == "initial"


def test_result_reports_its_own_corrections():
    """rxn_names is already resolved, so no annotate_from_model needed."""
    result = TunedKcatsResult(
        rxns=["r_slow", "r_untouched", "r_drift"],
        rxn_names=["slow enzyme", "untouched enzyme", "drifting enzyme"],
        old_kcat=np.array([1.0, 5.0, 2.0]),
        new_kcat=np.array([10.0, 5.0, 40.0]),
        source=["brenda", "custom", "okp"],
    )
    rows = result.corrections()

    # No leverage screen of its own, so rows fall back to fold-change order.
    assert [r.rxn_id for r in rows] == ["r_drift", "r_slow"]
    assert rows[0].name == "drifting enzyme"
    assert rows[1].source == "brenda"
    assert rows[1].fold_change == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Infeasible start: the same on every solver
# --------------------------------------------------------------------------- #

def _usable_solvers():
    solvers = []
    for name in ("glpk", "glpk_exact", "gurobi"):
        try:
            cobra.Model("probe").solver = name
        except Exception:
            continue
        solvers.append(name)
    return solvers


@pytest.fixture(params=_usable_solvers())
def solver(request):
    """Solver name; a solver that imports but is not licensed is skipped."""
    model = cobra.Model("probe")
    try:
        model.solver = request.param
        model.add_metabolites([cobra.Metabolite("m")])
        model.optimize()
    except Exception as exc:
        pytest.skip(f"{request.param} unusable: {exc}")
    return request.param


def test_infeasible_start_raises_on_every_solver(tmp_path, solver):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0, enzyme_mw=100.0)
    model.solver = solver
    model.reactions.get_by_id("biomass").lower_bound = 500.0
    assert model.optimize().status == "infeasible"
    with pytest.raises(ValueError, match="not optimal"):
        sensitivity_tuning(model)
