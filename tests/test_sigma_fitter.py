"""Tests for fit_sigma."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.kcat_tuning import (
    SigmaFitterResult,
    fit_sigma,
)


# --------------------------------------------------------------------------- #
# Tiny enzyme-constrained model fixture
# --------------------------------------------------------------------------- #

def _adapter(
    tmp_path: Path,
    *,
    p_tot: float = 1.0,
    f: float = 1.0,
    gr_exp: float = 0.5,
) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\n'
        f'org_name = "test"\n'
        f'p_tot = {p_tot}\n'
        f'f = {f}\n'
        f'gr_exp = {gr_exp}\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_pool_only_model(adapter: ModelAdapter) -> EcModel:
    """A model where bio_rxn flux is bounded only by prot_pool_exchange.

    Topology:
        prot_pool_exchange:  -> prot_pool          (UB controlled by sigma)
        biomass:        prot_pool ->                (1 unit pool per unit growth)

    So max growth = prot_pool_exchange.upper_bound = 1000 * p_tot * f * sigma.
    With p_tot=1, f=1: growth = 1000*sigma. The optimal sigma is
    growth_rate / 1000.
    """
    model = EcModel("toy", adapter=adapter)

    pool = cobra.Metabolite("prot_pool", compartment="c")
    model.add_metabolites([pool])

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0
    pool_ex.upper_bound = 1000.0

    bio = cobra.Reaction("biomass")
    bio.add_metabolites({pool: -1.0})
    bio.lower_bound = 0.0
    bio.upper_bound = 1000.0

    model.add_reactions([pool_ex, bio])
    model.objective = "biomass"

    model.ec = EcData(
        rxns=[],
        kcat=np.empty(0, dtype=float),
        source=[],
        notes=[],
        eccodes=[],
        rxn_enz_mat=sparse.csr_matrix((0, 0), dtype=float),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_n_sigma_steps_zero_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_pool_only_model(adapter)
    with pytest.raises(ValueError, match="n_sigma_steps"):
        fit_sigma(model, n_sigma_steps=0)


def test_no_adapter_no_defaults_raises(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_pool_only_model(adapter)
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        fit_sigma(model)


# --------------------------------------------------------------------------- #
# Best-sigma selection
# --------------------------------------------------------------------------- #

def test_best_sigma_close_to_target_growth(tmp_path):
    """With p_tot=1, f=1, growth = 1000*sigma, so for growth_rate=10 the
    exactly-optimal sigma is 0.01; the grid scan should land on it with
    zero error."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model)
    assert isinstance(result, SigmaFitterResult)
    # growth = 1000*sigma; for growth=10, sigma=0.01.
    assert result.sigma == pytest.approx(0.01, abs=1e-4)
    # The 0.01 grid contains the exactly-optimal sigma, so the error is zero.
    best_idx = int(np.argmin(result.error_grid))
    assert result.error_grid[best_idx] < 0.1


def test_bisect_matches_grid_optimum_with_fewer_solves(tmp_path):
    """method='bisect' finds the same optimum as the full grid using far
    fewer LP solves (growth is monotone in sigma)."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, method="bisect")
    assert result.sigma == pytest.approx(0.01, abs=1e-4)
    # ~log2 evaluations instead of the 100-point grid.
    assert len(result.sigma_grid) < 30
    pool_ub = model.reactions.get_by_id("prot_pool_exchange").upper_bound
    assert pool_ub == pytest.approx(1000.0 * result.sigma)


def test_bisect_invalid_method_raises(tmp_path):
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    with pytest.raises(ValueError, match="method"):
        fit_sigma(model, method="nope")


def test_growth_grid_scales_linearly_with_sigma(tmp_path):
    """For the toy model, growth = 1000*sigma."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, n_sigma_steps=10)
    expected = result.sigma_grid * 1000.0
    np.testing.assert_allclose(result.growth_grid, expected, rtol=1e-6)


def test_optimal_sigma_applied_to_model_at_end(tmp_path):
    """After fitting, the model is left at the optimal sigma found, not
    the last sigma tried during the search."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=50.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model)
    # Pool exchange ub should reflect optimal sigma:
    # ub = p_tot * f * sigma * 1000 = 1 * 1 * sigma * 1000 = 1000*sigma
    expected_ub = 1000.0 * result.sigma
    assert (
        model.reactions.get_by_id("prot_pool_exchange").upper_bound
        == pytest.approx(expected_ub)
    )


def test_model_at_optimal_sigma_reaches_growth_rate(tmp_path):
    """After fitting, the model's LP optimum matches growth_rate (within
    the grid resolution: here the grid contains the optimal sigma)."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=50.0)
    model = _build_pool_only_model(adapter)
    fit_sigma(model)
    sol = model.optimize()
    assert sol.objective_value == pytest.approx(50.0, abs=1e-2)


# --------------------------------------------------------------------------- #
# Defaults from adapter
# --------------------------------------------------------------------------- #

def test_defaults_pulled_from_adapter(tmp_path):
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    # Don't pass any explicit kwargs; everything should come from adapter.
    result = fit_sigma(model)
    assert result.sigma == pytest.approx(0.01, abs=1e-4)


def test_explicit_growth_rate_overrides_adapter(tmp_path):
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=999.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, growth_rate=10.0)
    assert result.sigma == pytest.approx(0.01, abs=1e-4)


def test_explicit_p_tot_and_f_override_adapter(tmp_path):
    adapter = _adapter(tmp_path, p_tot=999.0, f=999.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    # Override: p_tot=1, f=1 -> growth = 1000*sigma; sigma=0.01 for growth=10.
    result = fit_sigma(model, p_tot=1.0, f=1.0)
    assert result.sigma == pytest.approx(0.01, abs=1e-4)


# --------------------------------------------------------------------------- #
# Grid shape
# --------------------------------------------------------------------------- #

def test_grid_n_sigma_steps_default_is_100(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, method="grid")
    assert result.sigma_grid.shape == (100,)
    assert result.growth_grid.shape == (100,)
    assert result.error_grid.shape == (100,)


def test_custom_n_sigma_steps(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, n_sigma_steps=10, method="grid")
    assert result.sigma_grid.shape == (10,)
    np.testing.assert_allclose(
        result.sigma_grid,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )


def test_sigma_grid_starts_above_zero_ends_at_one(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, n_sigma_steps=20, method="grid")
    assert result.sigma_grid[0] > 0
    assert result.sigma_grid[-1] == pytest.approx(1.0)


def test_default_method_is_the_full_grid(tmp_path):
    """Default method is the grid of sigmaFitter.m: every one of the 100
    sigmas is evaluated."""
    adapter = _adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model)  # default == grid
    assert result.sigma == pytest.approx(0.01, abs=1e-12)
    assert len(result.sigma_grid) == 100
