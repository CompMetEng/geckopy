"""Tests for relax_enz_concs: the least total relaxation, in one LP."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import relax_enz_concs


def _adapter(tmp_path: Path, gr_exp: float) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\norg_name = "test"\ngr_exp = {gr_exp}\nbio_rxn = "biomass"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _two_routes(adapter, caps, pool=1000.0):
    """A -> B by enzyme E1 (2 mg per unit flux) or E2 (1 mg per unit); growth = flux to B.

    ``caps`` gives the measured concentration of each enzyme (mg).
    """
    model = EcModel("routes", adapter=adapter)
    mets = {n: cobra.Metabolite(n, compartment="c") for n in ("A", "B", "pool", "prot_E1", "prot_E2")}
    model.add_metabolites(list(mets.values()))
    spec = {
        "EX_A": ({"A": 1}, (0, 10)),
        "R1": ({"A": -1, "B": 1, "prot_E1": -2}, (0, 1000)),
        "R2": ({"A": -1, "B": 1, "prot_E2": -1}, (0, 1000)),
        "biomass": ({"B": -1}, (0, 1000)),
        "prot_pool_exchange": ({"pool": 1}, (0, pool)),
        "usage_prot_E1": ({"pool": -1, "prot_E1": 1}, (0, caps["E1"])),
        "usage_prot_E2": ({"pool": -1, "prot_E2": 1}, (0, caps["E2"])),
    }
    for rid, (st, bounds) in spec.items():
        r = cobra.Reaction(rid)
        r.add_metabolites({mets[m]: c for m, c in st.items()})
        r.bounds = bounds
        model.add_reactions([r])
    model.objective = "biomass"
    model.ec = EcData(
        rxns=["R1", "R2"], kcat=np.ones(2), source=[""] * 2, notes=[""] * 2, eccodes=[""] * 2,
        genes=["g1", "g2"], enzymes=["E1", "E2"], mw=np.full(2, 100.0), sequence=["", ""],
        concs=np.array([caps["E1"], caps["E2"]], dtype=float), rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    return model


def test_the_least_summed_fold_increase_reaches_the_target(tmp_path):
    # 8 units: x on E1 (2 mg each) and 8 - x on E2 (1 mg each), both capped at 2 mg.
    # Folds (x - 1)+ and (3 - x/2)+ sum to 2 + x/2 on [1, 6], 3 - x/2 below: least is 2.5 at x = 1,
    # E1 kept at 2 mg and E2 raised to 7 mg.
    model = _two_routes(_adapter(tmp_path, gr_exp=8.0), {"E1": 2.0, "E2": 2.0})
    result = relax_enz_concs(model)
    assert model.optimize().objective_value == pytest.approx(8.0)
    folds = model.reactions.usage_prot_E1.upper_bound / 2.0 - 1 + model.reactions.usage_prot_E2.upper_bound / 2.0 - 1
    assert folds == pytest.approx(2.5)
    assert model.reactions.usage_prot_E2.upper_bound == pytest.approx(7.0)
    assert "prot_pool" not in result.uniprot_ids


def test_enough_capacity_relaxes_nothing(tmp_path):
    model = _two_routes(_adapter(tmp_path, gr_exp=3.0), {"E1": 2.0, "E2": 2.0})
    result = relax_enz_concs(model)
    assert result.uniprot_ids == []
    assert model.reactions.usage_prot_E1.upper_bound == 2.0


def test_the_pool_grows_only_when_enzymes_cannot_help(tmp_path):
    model = _two_routes(_adapter(tmp_path, gr_exp=8.0), {"E1": 100.0, "E2": 100.0}, pool=4.0)
    result = relax_enz_concs(model)
    assert result.uniprot_ids == ["prot_pool"]
    assert model.reactions.prot_pool_exchange.upper_bound == pytest.approx(8.0)  # all on E2


def test_a_cap_raised_before_is_not_lowered(tmp_path):
    model = _two_routes(_adapter(tmp_path, gr_exp=8.0), {"E1": 2.0, "E2": 2.0})
    model.reactions.usage_prot_E2.upper_bound = 6.0  # raised earlier
    relax_enz_concs(model)
    assert model.reactions.usage_prot_E2.upper_bound >= 6.0
    assert model.optimize().objective_value == pytest.approx(8.0)


def test_an_unreachable_target_is_an_error_and_leaves_the_caps(tmp_path):
    model = _two_routes(_adapter(tmp_path, gr_exp=20.0), {"E1": 2.0, "E2": 2.0})  # supply caps growth at 10
    with pytest.raises(ValueError, match="not limited by protein"):
        relax_enz_concs(model)
    assert model.reactions.usage_prot_E1.upper_bound == 2.0
    assert model.reactions.prot_pool_exchange.upper_bound == 1000.0


def test_per_complex_raises_the_subunits_of_a_complex_together(tmp_path):
    from test_level_complex_concs import _adapter as adapter2, _complex_toy

    model = _complex_toy(adapter2(tmp_path, gr_exp=2.0))
    model.ec.concs = np.array([1.0, 1.0])
    model.reactions.usage_prot_X2.upper_bound = 1.0
    relax_enz_concs(model, per_complex=True)
    assert model.reactions.usage_prot_X1.upper_bound == pytest.approx(2.0)
    assert model.reactions.usage_prot_X2.upper_bound == pytest.approx(2.0)


def test_an_unknown_weighting_is_rejected(tmp_path):
    model = _two_routes(_adapter(tmp_path, gr_exp=8.0), {"E1": 2.0, "E2": 2.0})
    with pytest.raises(ValueError, match="weight"):
        relax_enz_concs(model, weight="max")
