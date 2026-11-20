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
# Source annotation
# --------------------------------------------------------------------------- #

def test_source_gets_tuning_status_with_prior(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    sensitivity_tuning(model, fold_change=10.0)
    assert model.ec.source[0] == "initial | tuned=sensitivity (prior=1)"


def test_notes_are_left_alone(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    notes = list(model.ec.notes)
    sensitivity_tuning(model, fold_change=10.0)
    assert list(model.ec.notes) == notes


def test_re_tuning_keeps_one_method_and_the_first_prior(tmp_path):
    """A second tuning pass on a tuned kcat does not repeat the method and
    keeps the kcat before the first tuning as prior."""
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    sensitivity_tuning(model, fold_change=10.0)
    assert model.ec.source[0] == "initial | tuned=sensitivity (prior=1)"

    sensitivity_tuning(
        model, desired_growth_rate=3000.0, fold_change=10.0,
    )
    assert model.ec.kcat[0] > 10.0
    assert model.ec.source[0] == "initial | tuned=sensitivity (prior=1)"


# --------------------------------------------------------------------------- #
# prot_to_ignore
# --------------------------------------------------------------------------- #

def test_evidence_of_the_tuned_reactions(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0)
    off = sensitivity_tuning(model, verbose=False)
    assert off.rxns and off.evidence is None
    model = _build_toy(adapter, initial_kcat=1.0)
    tuned = sensitivity_tuning(model, verbose=False, evidence=True)
    assert list(tuned.evidence["ec_rxn"]) == tuned.rxns
    assert tuned.evidence["kcat"].tolist() == tuned.new_kcat.tolist()
    assert tuned.evidence["prior"].tolist() == tuned.old_kcat.tolist()


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
    model.slim_optimize(error_value=float("nan"))
    assert model.solver.status == "infeasible"
    with pytest.raises(ValueError, match="not optimal"):
        sensitivity_tuning(model)


def test_infeasible_start_raises_when_gurobi_presolves_it(tmp_path):
    """Gurobi holds no primal values for an infeasible LP it presolved; the start check must not read them."""
    adapter = _adapter(tmp_path, gr_exp=300.0)
    model = _build_toy(adapter, initial_kcat=1.0, enzyme_mw=100.0)
    try:
        model.solver = "gurobi"
    except Exception as exc:
        pytest.skip(f"gurobi unusable: {exc}")
    model.solver.problem.Params.Presolve = 2
    model.reactions.get_by_id("biomass").lower_bound = 500.0
    with pytest.raises(ValueError, match="not optimal"):
        sensitivity_tuning(model)


def test_docstrings_name_the_standard_and_advanced_tuning():
    from geckopy.kcat_tuning import sensitivity_tuning
    from geckopy.kcat_tuning.evotune import cmaes_kcat_tuning

    assert "standard kcat tuning" in sensitivity_tuning.__doc__
    assert "advanced kcat tuning" in cmaes_kcat_tuning.__doc__


# --------------------------------------------------------------------------- #
# Tuning to a measured condition (growth and exchange fluxes)
# --------------------------------------------------------------------------- #

def _build_two_branches(
    adapter: ModelAdapter, *, mw_f: float = 200.0, uptake: float = 1000.0,
) -> EcModel:
    """A -> biomass through enzyme E, and A -> P (secreted) through enzyme F.

    Both draw on a 1 mg/gDW pool, with kcat 1 and a coefficient of
    ``mw / 3600`` mg/mmol per unit flux.
    """
    from geckopy.databases.flux_data import FluxData  # noqa: F401  (import check)

    model = EcModel("toy2", adapter=adapter)
    mets = {n: cobra.Metabolite(n, compartment="c") for n in
            ("A_e", "A_c", "P_c", "P_e", "prot_pool", "prot_E", "prot_F", "bio_met")}
    model.add_metabolites(list(mets.values()))

    def rxn(rid, stoich, lb, ub):
        r = cobra.Reaction(rid)
        r.add_metabolites({mets[k]: v for k, v in stoich.items()})
        r.bounds = (lb, ub)
        return r

    mw_e = 100.0
    model.add_reactions([
        rxn("EX_A", {"A_e": -1.0}, -uptake, 0.0),
        rxn("TR_A", {"A_e": -1.0, "A_c": 1.0}, 0.0, 1000.0),
        rxn("R", {"A_c": -1.0, "prot_E": -mw_e / 3600.0, "bio_met": 1.0}, 0.0, 1000.0),
        rxn("R2", {"A_c": -1.0, "prot_F": -mw_f / 3600.0, "P_c": 1.0}, 0.0, 1000.0),
        rxn("TR_P", {"P_c": -1.0, "P_e": 1.0}, 0.0, 1000.0),
        rxn("EX_P", {"P_e": -1.0}, 0.0, 1000.0),
        rxn("biomass", {"bio_met": -1.0}, 0.0, 1000.0),
        rxn("prot_pool_exchange", {"prot_pool": 1.0}, 0.0, 1.0),
        rxn("usage_prot_E", {"prot_pool": -1.0, "prot_E": 1.0}, 0.0, 1000.0),
        rxn("usage_prot_F", {"prot_pool": -1.0, "prot_F": 1.0}, 0.0, 1000.0),
    ])
    model.objective = "biomass"
    model.ec = EcData(
        rxns=["R", "R2"], kcat=np.array([1.0, 1.0]), source=["initial", "initial"],
        notes=["", ""], eccodes=["", ""], genes=["g_E", "g_F"],
        enzymes=["E", "F"], mw=np.array([mw_e, mw_f]), sequence=["", ""],
        concs=np.array([np.nan, np.nan]),
        rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    return model


def _condition(growth, p_flux, name="c1"):
    from geckopy.databases.flux_data import FluxData
    return FluxData(
        conds=[name], p_tot=np.array([np.nan]), gr_rate=np.array([growth]),
        exch_fluxes=np.array([[p_flux]]), exch_mets=["P"], exch_rxn_ids=["EX_P"],
    )


def test_growth_only_target_ignores_the_secreted_flux(tmp_path):
    """Growth 20 is reachable (the maximum is 36), so the growth-only
    tuning changes nothing, although growth 20 with P at 20 is not."""
    model = _build_two_branches(_adapter(tmp_path))
    result = sensitivity_tuning(model, desired_growth_rate=20.0)
    assert result.rxns == []
    assert result.status == "reached"


def test_condition_raises_the_kcat_of_the_enzyme_that_draws_most(tmp_path):
    model = _build_two_branches(_adapter(tmp_path))
    result = sensitivity_tuning(
        model, flux_data=_condition(20.0, 20.0), fold_change=10.0)
    # F has the larger molecular weight, so it draws most of the pool.
    assert result.rxns == ["R2"]
    assert model.ec.kcat.tolist() == [1.0, 10.0]
    assert result.status == "reached"
    with model:
        model.reactions.EX_P.bounds = (18.0, 22.0)
        model.reactions.biomass.lower_bound = 20.0
        assert model.slim_optimize(error_value=float("nan")) >= 20.0 - 1e-6


def test_condition_by_name_and_growth_override(tmp_path):
    model = _build_two_branches(_adapter(tmp_path))
    flux = _condition(1.0, 20.0, name="chemostat")
    result = sensitivity_tuning(
        model, flux_data=flux, condition="chemostat", desired_growth_rate=20.0)
    assert result.rxns == ["R2"]


def test_a_measured_flux_that_needs_no_tuning_leaves_the_kcats(tmp_path):
    model = _build_two_branches(_adapter(tmp_path))
    result = sensitivity_tuning(model, flux_data=_condition(10.0, 10.0))
    assert result.rxns == [] and result.status == "reached"


def test_tolerance_widens_what_counts_as_met(tmp_path):
    # Growth 20 with P 17.5: the pool needs 20*0.0278 + 17.5*0.0556 = 1.53.
    # With P within 40% (down to 12) the condition is met at 12 and 20: 1.22,
    # still over the pool, so the kcat of F is raised either way; at 60% (7)
    # growth 20 and P 7 need 0.94 and nothing changes.
    flux = _condition(20.0, 17.5)
    tight = _build_two_branches(_adapter(tmp_path))
    assert sensitivity_tuning(tight, flux_data=flux, flux_tolerance=0.1).rxns == ["R2"]
    loose = _build_two_branches(_adapter(tmp_path))
    result = sensitivity_tuning(loose, flux_data=flux, flux_tolerance=0.6)
    assert result.rxns == []


def test_per_flux_tolerance_overrides_the_default(tmp_path):
    flux = _condition(20.0, 17.5)
    model = _build_two_branches(_adapter(tmp_path))
    result = sensitivity_tuning(
        model, flux_data=flux, flux_tolerance=0.01, flux_tolerances={"EX_P": 0.6})
    assert result.rxns == []


def test_a_condition_no_kcat_can_meet_is_reported_unfixable(tmp_path):
    """Growth 20 needs 20 mmol/gDW/h of A, and the uptake is limited to 10."""
    model = _build_two_branches(_adapter(tmp_path), uptake=10.0)
    before = model.ec.kcat.copy()
    result = sensitivity_tuning(model, flux_data=_condition(20.0, 0.0))
    assert result.status == "unfixable"
    assert result.rxns == []
    assert np.array_equal(model.ec.kcat, before)


def test_the_condition_does_not_stay_applied_to_the_model(tmp_path):
    model = _build_two_branches(_adapter(tmp_path))
    sensitivity_tuning(model, flux_data=_condition(20.0, 20.0))
    assert model.reactions.EX_P.bounds == (0.0, 1000.0)
    assert model.reactions.biomass.bounds == (0.0, 1000.0)
    assert model.reactions.prot_pool_exchange.bounds == (0.0, 1.0)


def test_unknown_condition_name_raises(tmp_path):
    model = _build_two_branches(_adapter(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        sensitivity_tuning(model, flux_data=_condition(20.0, 20.0), condition="x")


def test_exchange_missing_from_the_model_raises(tmp_path):
    from geckopy.databases.flux_data import FluxData
    model = _build_two_branches(_adapter(tmp_path))
    flux = FluxData(
        conds=["c"], p_tot=np.array([np.nan]), gr_rate=np.array([20.0]),
        exch_fluxes=np.array([[1.0]]), exch_mets=["Q"], exch_rxn_ids=["EX_Q"])
    with pytest.raises(ValueError, match="not in the model"):
        sensitivity_tuning(model, flux_data=flux)


def test_status_is_reported_without_a_condition(tmp_path):
    model = _build_toy(_adapter(tmp_path, gr_exp=300.0))
    assert sensitivity_tuning(model, fold_change=10.0).status == "reached"
