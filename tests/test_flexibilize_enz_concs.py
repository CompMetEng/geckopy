"""Tests for flexibilize_enz_concs."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import (
    FlexEnzResult,
    flexibilize_enz_concs,
)


# --------------------------------------------------------------------------- #
# Model fixture
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path, *, gr_exp: float, bio_rxn: str) -> ModelAdapter:
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
    enzyme_ub: float = 1.0,
    measured_conc: float = 1.0,
) -> EcModel:
    """Tiny enzyme-constrained model where bio_rxn flux equals enzyme ub."""
    model = EcModel("toy", adapter=adapter)

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([A_e, A_c, B_c, pool, prot_E])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -1000.0
    EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0
    TR_A.upper_bound = 1000.0

    R_AB = cobra.Reaction("R_AB")
    R_AB.add_metabolites({A_c: -1.0, B_c: 1.0, prot_E: -1.0})
    R_AB.lower_bound = 0.0
    R_AB.upper_bound = 1000.0

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({B_c: -1.0})
    BIO.lower_bound = 0.0
    BIO.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0
    pool_ex.upper_bound = 1000.0

    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage.lower_bound = 0.0
    usage.upper_bound = enzyme_ub

    model.add_reactions([EX_A, TR_A, R_AB, BIO, pool_ex, usage])
    model.objective = "biomass"

    model.ec = EcData(
        rxns=["R_AB"],
        kcat=np.array([1.0]),
        source=[""],
        notes=[""],
        eccodes=[""],
        genes=["g_E"],
        enzymes=["E"],
        mw=np.array([100.0]),
        sequence=[""],
        concs=np.array([float(measured_conc)]),
        rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_no_measured_concentrations_raises(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    model.ec.concs = np.array([np.nan])
    with pytest.raises(ValueError, match="not measured|all-NaN"):
        flexibilize_enz_concs(model)


def test_no_adapter_no_exp_growth_raises(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    model.adapter = None
    with pytest.raises(ValueError, match="exp_growth"):
        flexibilize_enz_concs(model)


# --------------------------------------------------------------------------- #
# Happy path: limiting enzyme is relaxed
# --------------------------------------------------------------------------- #

def test_limiting_enzyme_relaxed_to_reach_exp_growth(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    # initial: enzyme ub = measured conc = 1.0; exp_growth = 5.0.
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)

    result = flexibilize_enz_concs(model)

    assert isinstance(result, FlexEnzResult)
    assert result.uniprot_ids == ["E"]
    # The refinement pass calibrates enzyme UBs to the soft band's low
    # edge (raven-gecko-parity#71: matches MATLAB's setParam('var', ...,
    # 0.5), a +/-0.25% band), not to exp_growth exactly -- so the UB
    # only needs to support ~4.9875, not the full 5.0.
    assert model.reactions.get_by_id("usage_prot_E").upper_bound >= 4.9875
    # Verify model reaches the band's low edge, not exp_growth exactly.
    sol = model.optimize()
    assert sol.objective_value == pytest.approx(4.9875, rel=1e-6)


def test_ec_concs_unchanged_after_flexibilization(tmp_path):
    """ec.concs is the original measurement; flexibilization should
    not modify it."""
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    flexibilize_enz_concs(model)
    assert model.ec.concs[0] == 1.0


def test_default_exp_growth_from_adapter(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=3.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    flexibilize_enz_concs(model)  # uses gr_exp = 3.0
    sol = model.optimize()
    # Band's low edge (raven-gecko-parity#71), not gr_exp exactly.
    assert sol.objective_value == pytest.approx(3.0 * 0.9975, rel=1e-6)


def test_explicit_exp_growth_overrides_adapter(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=999.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    flexibilize_enz_concs(model, exp_growth=4.0)
    sol = model.optimize()
    # Band's low edge (raven-gecko-parity#71), not exp_growth exactly.
    assert sol.objective_value == pytest.approx(4.0 * 0.9975, rel=1e-6)


# --------------------------------------------------------------------------- #
# No flexibilization needed
# --------------------------------------------------------------------------- #

def test_already_at_exp_growth_no_flexibilization(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=1.0, bio_rxn="biomass")
    # initial enzyme ub = 5 supports growth = 5; exp_growth = 1.
    model = _build_toy(adapter, enzyme_ub=5.0, measured_conc=5.0)
    result = flexibilize_enz_concs(model)
    # No relaxation needed; result should be empty.
    assert len(result.uniprot_ids) == 0


# --------------------------------------------------------------------------- #
# Iter limit
# --------------------------------------------------------------------------- #

def test_iter_per_enzyme_limit_warns_and_breaks(tmp_path, caplog):
    """If exp_growth is unreachable even after iter_per_enzyme
    relaxations, the function warns and stops."""
    import logging
    adapter = _adapter(tmp_path, gr_exp=1000.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    # Cap upstream so even infinite enzyme can't help: limit substrate.
    model.reactions.get_by_id("EX_A").lower_bound = -10.0
    with caplog.at_level(logging.WARNING):
        flexibilize_enz_concs(model, iter_per_enzyme=2)
    # Either iter limit hit OR pool branch tried; both produce log output.
    assert (
        "iter limit reached" in caplog.text
        or "protein pool" in caplog.text
        or "Maximum growth" in caplog.text
    )


# --------------------------------------------------------------------------- #
# Bio_rxn upper bound auto-raised
# --------------------------------------------------------------------------- #

def test_bio_rxn_upper_bound_raised_to_exp_growth(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    model.reactions.get_by_id("biomass").upper_bound = 0.5
    flexibilize_enz_concs(model, exp_growth=5.0)
    assert model.reactions.get_by_id("biomass").upper_bound >= 5.0


# --------------------------------------------------------------------------- #
# Result sorting
# --------------------------------------------------------------------------- #

def test_result_arrays_have_consistent_lengths(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    result = flexibilize_enz_concs(model)
    n = len(result.uniprot_ids)
    assert len(result.old_concs) == n
    assert len(result.flex_concs) == n
    assert len(result.ratio_incr) == n
    assert len(result.frequence) == n


def test_result_ratio_incr_sorted_descending(tmp_path):
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    result = flexibilize_enz_concs(model)
    ratios = list(result.ratio_incr)
    assert ratios == sorted(ratios, reverse=True)


def test_flex_concs_greater_than_old_for_returned_enzymes(tmp_path):
    """Enzymes in the result should have flex_concs > old_concs by
    definition (else they would have been excluded)."""
    adapter = _adapter(tmp_path, gr_exp=5.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    result = flexibilize_enz_concs(model)
    for old, flex in zip(result.old_concs, result.flex_concs):
        assert flex >= old


# --------------------------------------------------------------------------- #
# Infeasible start: the same on every solver
# --------------------------------------------------------------------------- #

def _usable_solvers():
    solvers = []
    for name in ("glpk", "glpk_exact", "gurobi", "scipy"):
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
        probe = cobra.Reaction("probe_rxn")
        probe.bounds = (0, 1)
        model.add_reactions([probe])
        model.optimize()
    except Exception as exc:
        pytest.skip(f"{request.param} unusable: {exc}")
    return request.param


def test_infeasible_start_returns_empty_result_on_every_solver(tmp_path, solver):
    adapter = _adapter(tmp_path, gr_exp=500.0, bio_rxn="biomass")
    model = _build_toy(adapter, enzyme_ub=1.0, measured_conc=1.0)
    model.solver = solver
    model.reactions.get_by_id("R_AB").lower_bound = 50.0
    assert model.optimize().status == "infeasible"
    result = flexibilize_enz_concs(model)
    assert result.uniprot_ids == []
    assert model.reactions.get_by_id("usage_prot_E").upper_bound == 1.0
    assert model.reactions.get_by_id("R_AB").lower_bound == 50.0
