"""Tests for ec_fseof (thin wrapper around raven_toolbox.analysis.fseof)."""
from pathlib import Path

import cobra
import logging
import numpy as np
import pandas as pd
import pytest
from raven_toolbox.analysis.fseof import FSEOFResult
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import ec_fseof


# --------------------------------------------------------------------------- #
# Tiny FSEOF-able ec model fixture
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path, *, bio_rxn: str = "biomass") -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\n'
        f'org_name = "test"\n'
        f'bio_rxn = "{bio_rxn}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_fseof_model(adapter: ModelAdapter) -> EcModel:
    """Substrate A is split between biomass (via R_BIO) and the
    production target (via R_PROD). Both routes consume an enzyme,
    drawn from a shared pool — the upper bound on the pool creates the
    tradeoff that makes FSEOF meaningful.

    EX_A:                A_e <-                    (uptake)
    TR_A:                A_e -> A_c
    R_BIO:    A_c + (1/100)*prot_E_BIO -> bio_met  (gpr: g_BIO)
    R_PROD:   A_c + (1/200)*prot_E_PROD -> prod_met (gpr: g_PROD)
    biomass:             bio_met ->                (objective)
    EX_PROD:             prod_met ->               (production target)
    usage_prot_E_BIO:    prot_pool -> prot_E_BIO
    usage_prot_E_PROD:   prot_pool -> prot_E_PROD
    prot_pool_exchange:  -> prot_pool              (small ub forces tradeoff)
    """
    model = EcModel("toy", adapter=adapter)

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    prod_met = cobra.Metabolite("prod_met", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_BIO = cobra.Metabolite("prot_E_BIO", compartment="c")
    prot_PROD = cobra.Metabolite("prot_E_PROD", compartment="c")
    model.add_metabolites([
        A_e, A_c, bio_met, prod_met, pool, prot_BIO, prot_PROD,
    ])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -100.0; EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0; TR_A.upper_bound = 1000.0

    R_BIO = cobra.Reaction("R_BIO")
    R_BIO.add_metabolites({A_c: -1.0, prot_BIO: -1/100, bio_met: 1.0})
    R_BIO.lower_bound = 0.0; R_BIO.upper_bound = 1000.0
    R_BIO.gene_reaction_rule = "g_BIO"

    R_PROD = cobra.Reaction("R_PROD")
    R_PROD.add_metabolites({A_c: -1.0, prot_PROD: -1/200, prod_met: 1.0})
    R_PROD.lower_bound = 0.0; R_PROD.upper_bound = 1000.0
    R_PROD.gene_reaction_rule = "g_PROD"

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.lower_bound = 0.0; BIO.upper_bound = 1000.0

    EX_PROD = cobra.Reaction("EX_PROD")
    EX_PROD.add_metabolites({prod_met: -1.0})
    EX_PROD.lower_bound = 0.0; EX_PROD.upper_bound = 1000.0

    usage_BIO = cobra.Reaction("usage_prot_E_BIO")
    usage_BIO.add_metabolites({pool: -1.0, prot_BIO: 1.0})
    usage_BIO.lower_bound = 0.0; usage_BIO.upper_bound = 1000.0

    usage_PROD = cobra.Reaction("usage_prot_E_PROD")
    usage_PROD.add_metabolites({pool: -1.0, prot_PROD: 1.0})
    usage_PROD.lower_bound = 0.0; usage_PROD.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0; pool_ex.upper_bound = 0.5  # forces tradeoff

    model.add_reactions([
        EX_A, TR_A, R_BIO, R_PROD, BIO, EX_PROD,
        usage_BIO, usage_PROD, pool_ex,
    ])
    model.objective = "biomass"

    n = 2
    g = 2
    mat = sparse.lil_matrix((n, g), dtype=float)
    mat[0, 0] = 1.0  # R_BIO uses E_BIO
    mat[1, 1] = 1.0  # R_PROD uses E_PROD
    model.ec = EcData(
        rxns=["R_BIO", "R_PROD"],
        kcat=np.array([1.0, 1.0]),
        source=["initial", "initial"],
        notes=["", ""],
        eccodes=["", ""],
        genes=["g_BIO", "g_PROD"],
        enzymes=["E_BIO", "E_PROD"],
        mw=np.array([100.0, 100.0]),
        sequence=["", ""],
        concs=np.array([np.nan, np.nan]),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_no_adapter_raises(tmp_path):
    """resolve_param fails when neither model.adapter nor bio_rxn is set."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        ec_fseof(model, "EX_PROD")


def test_explicit_bio_rxn_overrides_adapter(tmp_path):
    """Passing bio_rxn= explicitly bypasses the adapter lookup."""
    adapter = _adapter(tmp_path, bio_rxn="should_not_be_used")
    model = _build_fseof_model(adapter)
    # adapter says bio_rxn="should_not_be_used" (non-existent); explicit
    # arg = "biomass" should be used and the call must succeed.
    result = ec_fseof(model, "EX_PROD", bio_rxn="biomass", n_steps=4)
    assert isinstance(result, FSEOFResult)


def test_no_production_headroom_raises(tmp_path):
    """raven's fseof raises when the target reaction can't carry positive flux."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    # Block production entirely so EX_PROD's slim-optimise is 0.
    model.reactions.get_by_id("R_PROD").upper_bound = 0.0
    with pytest.raises(ValueError, match="cannot carry positive flux"):
        ec_fseof(model, "EX_PROD")


# --------------------------------------------------------------------------- #
# Carbon-source sanity-check warning
# --------------------------------------------------------------------------- #

def test_cs_rxn_warns_when_under_constrained(tmp_path, caplog):
    """If cs_rxn's lower bound is more negative than its uptake at
    biomass-max, emit a warning so the user can tighten it."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    # EX_A lb = -100 but biomass-max uptake is much smaller, so we warn.
    with caplog.at_level(logging.WARNING, logger="geckopy.utilities.ec_fseof"):
        ec_fseof(model, "EX_PROD", cs_rxn="EX_A", n_steps=4)
    assert any("carbon source" in r.getMessage() for r in caplog.records)


def test_cs_rxn_silent_when_tightly_bounded(tmp_path, caplog):
    """When cs_rxn lb equals (or is above) the biomass-max uptake, no warning."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    # First find what biomass-max actually uses, then set lb to that exact value.
    with model:
        model.objective = "biomass"
        sol = model.optimize()
        uptake = float(sol.fluxes["EX_A"])
    # uptake is negative (uptake convention); set lb equal so lb >= cs_flux.
    model.reactions.get_by_id("EX_A").lower_bound = uptake
    with caplog.at_level(logging.WARNING, logger="geckopy.utilities.ec_fseof"):
        ec_fseof(model, "EX_PROD", cs_rxn="EX_A", n_steps=4)
    assert not any("carbon source" in r.getMessage() for r in caplog.records)


def test_no_cs_rxn_no_warning(tmp_path, caplog):
    """Omitting cs_rxn skips the check entirely, even when EX_A is loose."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    with caplog.at_level(logging.WARNING, logger="geckopy.utilities.ec_fseof"):
        ec_fseof(model, "EX_PROD", n_steps=4)
    assert not any("carbon source" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Result shape (raven-toolbox's FSEOFResult)
# --------------------------------------------------------------------------- #

def test_returns_raven_fseof_result(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    assert isinstance(result, FSEOFResult)
    assert isinstance(result.scan, pd.DataFrame)
    assert isinstance(result.enforced, list)
    assert isinstance(result.targets, pd.DataFrame)


def test_enforced_levels_count_matches_n_steps_when_feasible(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    # raven may truncate if levels become infeasible; the toy model has
    # headroom so all 6 should succeed.
    assert len(result.enforced) == 6


def test_targets_have_expected_columns(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    expected_cols = {
        "reaction", "name", "subsystem", "gene_reaction_rule", "genes",
        "target_type", "slope", "correlation", "initial_flux",
        "final_flux", "score",
    }
    assert expected_cols.issubset(set(result.targets.columns))


# --------------------------------------------------------------------------- #
# Target identification (ec-specific behaviour: amplify on R_PROD)
# --------------------------------------------------------------------------- #

def test_r_prod_classified_as_amplify(tmp_path):
    """As the enforced EX_PROD flux rises, R_PROD's flux rises with it;
    raven's regression-based selection labels that ``amplify``."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=8)
    rprod = result.targets[result.targets["reaction"] == "R_PROD"]
    assert len(rprod) == 1, "R_PROD must be classified as a target"
    assert rprod.iloc[0]["target_type"] == "amplify"


def test_r_bio_classified_as_knockdown(tmp_path):
    """R_BIO's flux falls as biomass loses budget to the enforced
    product flux — raven labels that ``knockdown`` (or ``knockout``
    when the fall is to ~0)."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=8)
    rbio = result.targets[result.targets["reaction"] == "R_BIO"]
    assert len(rbio) == 1
    assert rbio.iloc[0]["target_type"] in ("knockdown", "knockout")


# --------------------------------------------------------------------------- #
# Scan floor anchored to biomass-optimal target flux
# --------------------------------------------------------------------------- #

def test_scan_floor_anchored_to_biomass_optimal_flux(tmp_path):
    """The scan's floor is the production-target's flux at biomass-optimum,
    not a fixed fraction of the theoretical maximum: forcing EX_PROD's
    lower bound to 5 makes that its biomass-optimal flux too (nothing
    rewards producing more), so the first enforced level must be 5, not
    raven-toolbox's default floor of target_max/n_steps."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    model.reactions.get_by_id("EX_PROD").lower_bound = 5.0
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    assert result.enforced[0] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Candidate set restricted to gene-associated, non-standard reactions
# --------------------------------------------------------------------------- #

def test_ungated_reaction_excluded_from_targets_and_scan(tmp_path):
    """EX_PROD itself has no GPR, so even though its flux trivially and
    perfectly tracks the enforced level, it must not appear as a target
    or in the reported scan -- matching ecFSEOF.m's restriction to
    gene-associated, non-standard reactions."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    assert "EX_PROD" not in result.scan.index
    assert "EX_PROD" not in set(result.targets["reaction"])


def test_standard_gene_reaction_excluded(tmp_path):
    """A reaction whose only GPR is the 'standard' placeholder pseudogene
    (added by assign_standard_kcat for un-annotated reactions) is not a
    real engineering target and must be excluded, matching ecFSEOF.m."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    model.reactions.get_by_id("R_BIO").gene_reaction_rule = "standard"
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    assert "R_BIO" not in result.scan.index
    assert "R_BIO" not in set(result.targets["reaction"])


# --------------------------------------------------------------------------- #
# usage_prot_* filtered out of scan + targets
# --------------------------------------------------------------------------- #

def test_usage_prot_rxns_dropped_from_scan(tmp_path):
    """The wrapper's job: ec-specific filtering of protein-pool plumbing
    out of raven's full per-reaction scan."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    for rid in result.scan.index:
        assert not rid.startswith("usage_prot_"), (
            f"usage_prot_* rows must be filtered from scan; saw {rid}"
        )


def test_usage_prot_rxns_dropped_from_targets(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    for rid in result.targets["reaction"]:
        assert not rid.startswith("usage_prot_"), (
            f"usage_prot_* rows must be filtered from targets; saw {rid}"
        )


def test_gene_targets_property_recomputes_off_filtered_targets(tmp_path):
    """raven's FSEOFResult.gene_targets is a @property over .targets;
    once we filter usage_prot_* from targets, the per-gene rollup
    inherits the filter automatically."""
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=6)
    # The toy model's only genes on real reactions are g_BIO and g_PROD;
    # usage_prot_* reactions have no GPR, so the rollup is clean.
    assert set(result.gene_targets["gene"]).issubset({"g_BIO", "g_PROD"})


# --------------------------------------------------------------------------- #
# Convenience slicers (raven properties survive filtering)
# --------------------------------------------------------------------------- #

def test_amplification_slice_works(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=8)
    amp = result.amplification
    assert isinstance(amp, pd.DataFrame)
    assert "R_PROD" in list(amp["reaction"])


def test_knockout_slice_works(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_fseof_model(adapter)
    result = ec_fseof(model, "EX_PROD", n_steps=8)
    ko = result.knockout  # raven calls this slice "knockout" but it includes knockdown
    assert isinstance(ko, pd.DataFrame)
    assert "R_BIO" in list(ko["reaction"])


# --------------------------------------------------------------------------- #
# ecFSEOF.m's target order and fixed target flux
# --------------------------------------------------------------------------- #

def test_targets_list_reaction_targets_before_transport_targets():
    """A reaction whose metabolite names repeat (not counting prot_*) is a transport target;
    the score order is kept inside each group."""
    import importlib

    import pandas as pd

    module = importlib.import_module("geckopy.utilities.ec_fseof")
    model = cobra.Model("m")
    mets = {
        "a_c": cobra.Metabolite("a_c", name="a", compartment="c"),
        "a_e": cobra.Metabolite("a_e", name="a", compartment="e"),
        "b_c": cobra.Metabolite("b_c", name="b", compartment="c"),
        "p1": cobra.Metabolite("prot_E1", name="prot_E1", compartment="c"),
        "p2": cobra.Metabolite("prot_E2", name="prot_E2", compartment="c"),
    }
    transport = cobra.Reaction("TRANSPORT")
    transport.add_metabolites({mets["a_e"]: -1.0, mets["a_c"]: 1.0, mets["p1"]: -0.1})
    first = cobra.Reaction("FIRST")
    first.add_metabolites({mets["a_c"]: -1.0, mets["b_c"]: 1.0, mets["p2"]: -0.1})
    second = cobra.Reaction("SECOND")
    second.add_metabolites({mets["b_c"]: -1.0, mets["p1"]: -0.1, mets["p2"]: -0.1})
    model.add_reactions([transport, first, second])
    result = FSEOFResult(
        scan=pd.DataFrame(),
        enforced=[1.0],
        targets=pd.DataFrame({"reaction": ["TRANSPORT", "FIRST", "SECOND"], "score": [3.0, 2.0, 1.0]}),
    )

    grouped = module._group_transport_targets(result, model).targets

    assert list(grouped["reaction"]) == ["FIRST", "SECOND", "TRANSPORT"]
    assert list(grouped["transport"]) == [False, False, True]


def test_ec_fseof_fixes_the_target_flux_at_each_level(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("geckopy.utilities.ec_fseof")
    seen = {}
    real = module.fseof

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fseof", spy)
    result = ec_fseof(_build_fseof_model(_adapter(tmp_path)), "EX_PROD", n_steps=4)
    assert seen["equality"] is True
    assert "transport" in result.targets.columns


def test_scores_equal_to_nine_digits_keep_model_order():
    """Scores that differ only below 9 significant digits count as equal and are listed in
    model order."""
    import importlib

    import pandas as pd

    module = importlib.import_module("geckopy.utilities.ec_fseof")
    model = cobra.Model("m")
    met = cobra.Metabolite("a_c", name="a", compartment="c")
    reactions = []
    for rid in ("R1", "R2", "R3"):
        rxn = cobra.Reaction(rid)
        rxn.add_metabolites({met: -1.0})
        reactions.append(rxn)
    model.add_reactions(reactions)
    result = FSEOFResult(
        scan=pd.DataFrame(),
        enforced=[1.0],
        targets=pd.DataFrame({"reaction": ["R3", "R2", "R1"], "score": [2.0 + 1e-12, 2.0, 2.5]}),
    )

    grouped = module._group_transport_targets(result, model).targets

    assert list(grouped["reaction"]) == ["R1", "R2", "R3"]
