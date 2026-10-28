"""fit_sigma and sensitivity_tuning follow GECKO's sigmaFitter and sensitivityTuning."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter, fit_sigma, make_ec_model, sensitivity_tuning
from geckopy.ec_model.ec_data import EcData
from geckopy.ec_model.pipeline import apply_kcat_constraints, set_prot_pool_size

from test_sigma_fitter import _adapter as _sigma_adapter
from test_sigma_fitter import _build_pool_only_model

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


# --------------------------------------------------------------------------- #
# fit_sigma
# --------------------------------------------------------------------------- #

def test_default_scan_is_the_grid_of_sigma_fitter(tmp_path):
    """growth = 1000 * sigma; a target of 10.5 lies between the grid points 0.01 and 0.02."""
    adapter = _sigma_adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.5)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model)
    assert len(result.sigma_grid) == 100
    assert result.sigma == pytest.approx(0.01, abs=1e-12)


def test_bisect_is_still_selectable(tmp_path):
    adapter = _sigma_adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.5)
    model = _build_pool_only_model(adapter)
    result = fit_sigma(model, method="bisect")
    assert result.sigma == pytest.approx(0.0105, abs=1e-5)
    assert len(result.sigma_grid) < 30


def _capped_model(tmp_path, *, gr_exp, cap=30.0):
    """Growth is min(1000 * sigma, cap): a second substrate, limited to ``cap``,
    is consumed by the biomass reaction too."""
    adapter = _sigma_adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=gr_exp)
    model = _build_pool_only_model(adapter)
    substrate = cobra.Metabolite("cap_met", compartment="c")
    supply = cobra.Reaction("EX_cap")
    supply.add_metabolites({substrate: 1.0})
    supply.bounds = (0.0, cap)
    model.add_reactions([supply])
    model.reactions.get_by_id("biomass").add_metabolites({substrate: -1.0})
    return model


def test_unreachable_target_gives_the_start_of_the_growth_plateau(tmp_path):
    """Growth is min(1000 * sigma, 30); a target of 100 is out of reach, so the
    first grid point with the plateau's growth wins, as in sigmaFitter.m."""
    model = _capped_model(tmp_path, gr_exp=100.0)
    result = fit_sigma(model)
    assert result.sigma == pytest.approx(0.03, abs=1e-12)


def test_bisect_finds_the_start_of_the_plateau_for_an_unreachable_target(tmp_path):
    model = _capped_model(tmp_path, gr_exp=100.0)
    result = fit_sigma(model, method="bisect")
    assert result.sigma == pytest.approx(0.03, abs=1e-5)


def test_objective_bounds_are_opened_and_stay_open(tmp_path):
    """sigmaFitter.m sets lb 0 and ub 1000 on the objective reaction for good."""
    adapter = _sigma_adapter(tmp_path, p_tot=1.0, f=1.0, gr_exp=10.0)
    model = _build_pool_only_model(adapter)
    biomass = model.reactions.get_by_id("biomass")
    biomass.bounds = (5.0, 5.0)
    result = fit_sigma(model)
    assert biomass.bounds == (0.0, 1000.0)
    assert result.sigma == pytest.approx(0.01, abs=1e-12)


# --------------------------------------------------------------------------- #
# sensitivity_tuning, full model
# --------------------------------------------------------------------------- #

def _toy_adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\ngr_exp = 0.5\nbio_rxn = "biomass"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _chain(adapter, kcats, mws=(3600.0, 3600.0), pool=1.0) -> EcModel:
    """A -> B (R1, enzyme E1) -> biomass (R2, enzyme E2), both drawing on one pool.

    The pool cost of one unit of flux through Ri is mw_i / (kcat_i * 3600)."""
    model = EcModel("chain", adapter=adapter)
    mets = {name: cobra.Metabolite(name, compartment="c") for name in
            ("A_e", "A_c", "B_c", "pool", "prot_E1", "prot_E2", "bio_met")}
    mets["pool"].id = "prot_pool"
    model.add_metabolites(list(mets.values()))

    def rxn(rid, stoich, lb, ub):
        r = cobra.Reaction(rid)
        r.add_metabolites({mets[k]: v for k, v in stoich.items()})
        r.bounds = (lb, ub)
        return r

    c1 = mws[0] / (kcats[0] * 3600.0)
    c2 = mws[1] / (kcats[1] * 3600.0)
    model.add_reactions([
        rxn("EX_A", {"A_e": -1.0}, -1000.0, 0.0),
        rxn("TR_A", {"A_e": -1.0, "A_c": 1.0}, 0.0, 1000.0),
        rxn("R1", {"A_c": -1.0, "prot_E1": -c1, "B_c": 1.0}, 0.0, 1000.0),
        rxn("R2", {"B_c": -1.0, "prot_E2": -c2, "bio_met": 1.0}, 0.0, 1000.0),
        rxn("biomass", {"bio_met": -1.0}, 0.0, 1000.0),
        rxn("prot_pool_exchange", {"pool": 1.0}, 0.0, pool),
        rxn("usage_prot_E1", {"pool": -1.0, "prot_E1": 1.0}, 0.0, 1000.0),
        rxn("usage_prot_E2", {"pool": -1.0, "prot_E2": 1.0}, 0.0, 1000.0),
    ])
    model.objective = "biomass"
    model.ec = EcData(
        rxns=["R1", "R2"],
        kcat=np.array(kcats, dtype=float),
        source=["initial", "initial"],
        notes=["", ""],
        eccodes=["", ""],
        genes=["g1", "g2"],
        enzymes=["E1", "E2"],
        mw=np.array(mws, dtype=float),
        sequence=["", ""],
        concs=np.array([np.nan, np.nan]),
        rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    return model


def test_tuned_reactions_are_listed_in_model_order(tmp_path):
    """R2 costs more pool, so it is tuned first; the result lists R1 before R2."""
    model = _chain(_toy_adapter(tmp_path), kcats=[1.0, 0.5])
    result = sensitivity_tuning(model, desired_growth_rate=1.0, verbose=False)
    assert result.rxns == ["R1", "R2"]
    np.testing.assert_allclose(result.old_kcat, [1.0, 0.5])
    np.testing.assert_allclose(result.new_kcat, [10.0, 5.0])


def test_rerun_reports_the_kcat_and_source_of_the_input_model(tmp_path):
    model = _chain(_toy_adapter(tmp_path), kcats=[1.0, 0.5])
    sensitivity_tuning(model, desired_growth_rate=1.0, verbose=False)
    result = sensitivity_tuning(model, desired_growth_rate=30.0, verbose=False)
    assert result.rxns
    assert not np.isnan(result.old_kcat).any()
    assert all(src == "sensitivityTuning" for src in result.source)
    # The values the second run started from are the tuned ones of the first.
    assert set(result.old_kcat) <= {10.0, 5.0}


@pytest.mark.parametrize("value, text", [
    (10.0, "10"),
    (float(np.pi), "3.1416"),
    (0.5, "0.5"),
    (123456.789, "123456.789"),
    (1e7, "10000000"),
    (float("nan"), "NaN"),
])
def test_num2str_prints_a_scalar_as_matlab_does(value, text):
    from geckopy.kcat_tuning.sensitivity_tuning import _num2str
    assert _num2str(value) == text


def test_notes_hold_the_pre_tune_kcat_as_num2str_prints_it(tmp_path):
    model = _chain(_toy_adapter(tmp_path), kcats=[10.0, float(np.pi)])
    sensitivity_tuning(model, desired_growth_rate=30.0, verbose=False)
    assert "preTuneKcat=10 |" in model.ec.notes[0]
    assert "preTuneKcat=3.1416 |" in model.ec.notes[1]


def test_a_tuned_kcat_has_no_upper_limit(tmp_path):
    """kcat 2e6 and mw 7.2e9 give growth 1; a target of 100 needs 2e8, past the old 1e7 cap."""
    model = _chain(_toy_adapter(tmp_path), kcats=[2e6, 1e12], mws=(7.2e9, 3600.0))
    result = sensitivity_tuning(model, desired_growth_rate=99.0, verbose=False)
    assert result.rxns == ["R1"]
    assert result.new_kcat[0] == pytest.approx(2e8)


def test_prot_to_ignore_matches_without_regard_to_case(tmp_path):
    model = _chain(_toy_adapter(tmp_path), kcats=[1.0, 0.5])
    result = sensitivity_tuning(
        model, desired_growth_rate=1.0, prot_to_ignore=["e2"], verbose=False,
    )
    # E2 is ignored, so only R1 is ever tuned.
    assert result.rxns == ["R1"]


def test_prot_to_ignore_with_an_unknown_id_raises(tmp_path):
    model = _chain(_toy_adapter(tmp_path), kcats=[1.0, 0.5])
    with pytest.raises(ValueError, match="prot_to_ignore"):
        sensitivity_tuning(
            model, desired_growth_rate=1.0, prot_to_ignore=["NOPE"], verbose=False,
        )


# --------------------------------------------------------------------------- #
# sensitivity_tuning, gecko-light model
# --------------------------------------------------------------------------- #

def _light_model(kcats):
    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    conv = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    for rxn_id in ("R2", "R4"):
        conv.reactions.get_by_id(rxn_id).bounds = (0.0, 0.0)
    model = make_ec_model(conv, adapter, gecko_light=True)
    model.adapter = adapter
    for i, ec_id in enumerate(model.ec.rxns):
        base = str(ec_id).split("_", 1)[1]
        if base in kcats:
            model.ec.kcat[i] = float(kcats[base])
    model.objective = "R5"
    apply_kcat_constraints(model)
    set_prot_pool_size(model)
    return model


def test_gecko_light_model_is_tuned():
    model = _light_model({"R3": 0.05, "R5": 200.0})
    before = float(model.optimize().objective_value)
    kcat_before = np.array(model.ec.kcat, dtype=float)
    source_before = list(model.ec.source)
    result = sensitivity_tuning(
        model, desired_growth_rate=30.0, bio_rxn="R5", verbose=False,
    )
    assert result.rxns
    assert float(model.optimize().objective_value) > before
    # ec.source is left as it was on a light model, as in sensitivityTuning.m.
    assert list(model.ec.source) == source_before
    # Every isozyme row of a tuned reaction was multiplied by the same power of 10.
    for rxn_id in result.rxns:
        rows = [i for i, ec_id in enumerate(model.ec.rxns)
                if str(ec_id).split("_", 1)[1] == rxn_id and kcat_before[i] > 0]
        ratios = [model.ec.kcat[i] / kcat_before[i] for i in rows]
        assert ratios and np.allclose(ratios, ratios[0])
        assert ratios[0] >= 10.0 - 1e-9
