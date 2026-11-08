"""Tests for remove_standard_kcat."""
from pathlib import Path

import cobra
import numpy as np
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.databases import UniprotDB
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import assign_standard_kcat, remove_standard_kcat


# --------------------------------------------------------------------------- #
# Fixture builders (shared with test_assign_standard_kcat)
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path, *, enzyme_comp: str = "c") -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        f'conv_gem = "dummy.xml"\norg_name = "test"\nenzyme_comp = "{enzyme_comp}"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _uniprot(mws: list[float]) -> UniprotDB:
    n = len(mws)
    return UniprotDB(
        ids=[f"P{i:03d}" for i in range(n)],
        genes=[f"g{i}" for i in range(n)],
        eccodes=["1.1.1.1"] * n,
        mw=np.array(mws, dtype=float),
        sequences=[""] * n,
    )


def _build_model_with_pool(
    adapter: ModelAdapter,
    rxn_specs: list[tuple[str, list[tuple[str, float, str, str]], str | None]],
    *,
    ec_rxns: list[str],
    ec_kcats: list[float],
    ec_genes: list[str] | None = None,
    ec_enzymes: list[str] | None = None,
    ec_mws: list[float] | None = None,
    ec_sources: list[str] | None = None,
    rxn_to_ec_genes: list[list[int]] | None = None,
    gecko_light: bool = False,
    add_prot_pool: bool = True,
) -> EcModel:
    ec_genes = ec_genes or []
    ec_enzymes = ec_enzymes or []
    ec_mws = ec_mws or []
    ec_sources = ec_sources or [""] * len(ec_rxns)
    rxn_to_ec_genes = rxn_to_ec_genes or [[] for _ in ec_rxns]

    model = EcModel("test", adapter=adapter, gecko_light=gecko_light)
    mets: dict[str, cobra.Metabolite] = {}
    for _, met_list, _ in rxn_specs:
        for met_id, _, met_name, comp in met_list:
            if met_id not in mets:
                m = cobra.Metabolite(met_id, compartment=comp)
                m.name = met_name
                mets[met_id] = m
    if add_prot_pool and not gecko_light and "prot_pool" not in mets:
        pm = cobra.Metabolite("prot_pool", compartment="c")
        pm.name = "prot_pool"
        mets["prot_pool"] = pm
    if mets:
        model.add_metabolites(list(mets.values()))

    for rxn_id, met_list, gpr in rxn_specs:
        rxn = cobra.Reaction(rxn_id)
        rxn.lower_bound = 0.0
        rxn.upper_bound = 1000.0
        rxn.add_metabolites({mets[m]: c for m, c, _, _ in met_list})
        if gpr:
            rxn.gene_reaction_rule = gpr
        model.add_reactions([rxn])

    n = len(ec_rxns)
    g = len(ec_genes)
    mat = sparse.lil_matrix((n, g), dtype=float)
    for i, gene_indices in enumerate(rxn_to_ec_genes):
        for j in gene_indices:
            mat[i, j] = 1.0

    model.ec = EcData(
        gecko_light=gecko_light,
        rxns=list(ec_rxns),
        kcat=np.array(ec_kcats, dtype=float),
        source=list(ec_sources),
        notes=[""] * n,
        eccodes=[""] * n,
        genes=list(ec_genes),
        enzymes=list(ec_enzymes),
        mw=np.array(ec_mws, dtype=float),
        sequence=[""] * g,
        concs=np.full(g, np.nan, dtype=float),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


# --------------------------------------------------------------------------- #
# Trivial cases / idempotency
# --------------------------------------------------------------------------- #

def test_no_standard_present_is_noop(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1")],
        ec_rxns=["r1"], ec_kcats=[5.0],
        ec_genes=["g1"], ec_enzymes=["g1"], ec_mws=[100.0],
        rxn_to_ec_genes=[[0]],
    )
    snapshot = {
        "ec_rxns": list(model.ec.rxns),
        "ec_enzymes": list(model.ec.enzymes),
        "n_mets": len(model.metabolites),
        "n_rxns": len(model.reactions),
    }
    remove_standard_kcat(model)
    assert list(model.ec.rxns) == snapshot["ec_rxns"]
    assert list(model.ec.enzymes) == snapshot["ec_enzymes"]
    assert len(model.metabolites) == snapshot["n_mets"]
    assert len(model.reactions) == snapshot["n_rxns"]


def test_empty_ec_is_noop(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1")],
        ec_rxns=[], ec_kcats=[],
    )
    remove_standard_kcat(model)
    assert model.ec.n_rxns == 0


def test_idempotent_double_call(tmp_path):
    """Running remove twice yields the same result as running it once."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    remove_standard_kcat(model)
    snap = {
        "ec_rxns": list(model.ec.rxns),
        "ec_enzymes": list(model.ec.enzymes),
        "n_mets": len(model.metabolites),
        "n_rxns": len(model.reactions),
    }
    remove_standard_kcat(model)
    assert list(model.ec.rxns) == snap["ec_rxns"]
    assert list(model.ec.enzymes) == snap["ec_enzymes"]
    assert len(model.metabolites) == snap["n_mets"]
    assert len(model.reactions) == snap["n_rxns"]


# --------------------------------------------------------------------------- #
# Standard pseudoenzyme removal
# --------------------------------------------------------------------------- #

def test_standard_enzyme_removed_from_ec(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "standard" in model.ec.enzymes
    remove_standard_kcat(model)
    assert "standard" not in model.ec.enzymes
    assert "standard" not in model.ec.genes


def test_standard_pseudoenzyme_rxns_removed_from_ec(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "r1" in model.ec.rxns
    remove_standard_kcat(model)
    assert "r1" not in model.ec.rxns


def test_topology_removed(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "prot_standard" in {m.id for m in model.metabolites}
    assert "usage_prot_standard" in {r.id for r in model.reactions}
    assert "standard" in {g.id for g in model.genes}

    remove_standard_kcat(model)
    assert "prot_standard" not in {m.id for m in model.metabolites}
    assert "usage_prot_standard" not in {r.id for r in model.reactions}
    assert "standard" not in {g.id for g in model.genes}


def test_rxn_enz_mat_shape_after_removal(tmp_path):
    """The rxn_enz_mat must shrink by one column AND any rows that
    pointed to the standard pseudoenzyme."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], None)],
        ec_rxns=["r1"], ec_kcats=[5.0],
        ec_genes=["g1"], ec_enzymes=["g1"], ec_mws=[100.0],
        rxn_to_ec_genes=[[0]],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    # After get: 2 rxns (r1, r2), 2 enzymes (g1, standard).
    assert model.ec.rxn_enz_mat.shape == (2, 2)

    remove_standard_kcat(model)
    # After remove: 1 rxn (r1), 1 enzyme (g1). r2 was added by get
    # and pointed to standard, so it gets removed.
    assert model.ec.rxn_enz_mat.shape == (1, 1)


# --------------------------------------------------------------------------- #
# fill_zero_kcat reset
# --------------------------------------------------------------------------- #

def test_filled_zero_kcat_reset_to_zero(tmp_path):
    """A reaction that had its kcat filled by fill_zero_kcat=True must
    have its kcat reset to 0 and source cleared, but stay in ec.rxns."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g2")],
        ec_rxns=["r1", "r2"],
        ec_kcats=[10.0, 0.0],   # r2 has zero kcat
        ec_genes=["g1", "g2"], ec_enzymes=["g1", "g2"], ec_mws=[100.0, 100.0],
        rxn_to_ec_genes=[[0], [1]],
    )
    assign_standard_kcat(model, _uniprot([100.0]), fill_zero_kcat=True)
    r2_idx = model.ec.rxns.index("r2")
    assert model.ec.source[r2_idx] == "standard"
    assert model.ec.kcat[r2_idx] > 0

    remove_standard_kcat(model)
    # r2 still in ec.rxns (linked to real g2, not standard pseudoenzyme).
    r2_idx = model.ec.rxns.index("r2")
    assert model.ec.kcat[r2_idx] == 0.0
    assert model.ec.source[r2_idx] == ""


def test_tuned_standard_kcat_is_kept(tmp_path):
    """A standard kcat with a tuning status was changed on purpose: it is
    not reset, while one with the bare origin is."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g2"),
         ("r3", [("C", -1.0, "gamma", "c"), ("D", 1.0, "delta", "c")], "g3")],
        ec_rxns=["r1", "r2", "r3"],
        ec_kcats=[10.0, 0.0, 0.0],
        ec_genes=["g1", "g2", "g3"], ec_enzymes=["g1", "g2", "g3"],
        ec_mws=[100.0, 100.0, 100.0],
        rxn_to_ec_genes=[[0], [1], [2]],
    )
    assign_standard_kcat(model, _uniprot([100.0]), fill_zero_kcat=True)
    r2_idx = model.ec.rxns.index("r2")
    r3_idx = model.ec.rxns.index("r3")
    tuned = "standard | tuned=cmaes (prior=7.5)"
    model.ec.source[r2_idx] = tuned
    kcat_r2 = float(model.ec.kcat[r2_idx])

    remove_standard_kcat(model)

    r2_idx = model.ec.rxns.index("r2")
    r3_idx = model.ec.rxns.index("r3")
    assert model.ec.source[r2_idx] == tuned
    assert model.ec.kcat[r2_idx] == kcat_r2
    assert model.ec.source[r3_idx] == ""
    assert model.ec.kcat[r3_idx] == 0.0


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #

def test_round_trip_restores_original_state(tmp_path):
    """assign_standard_kcat -> remove_standard_kcat returns the model
    to its original state (modulo S-matrix coefficients)."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], None)],
        ec_rxns=["r1"], ec_kcats=[5.0],
        ec_genes=["g1"], ec_enzymes=["g1"], ec_mws=[100.0],
        rxn_to_ec_genes=[[0]],
    )
    snapshot = {
        "ec_rxns": list(model.ec.rxns),
        "ec_kcat": model.ec.kcat.copy(),
        "ec_source": list(model.ec.source),
        "ec_enzymes": list(model.ec.enzymes),
        "ec_genes": list(model.ec.genes),
        "ec_mw": model.ec.mw.copy(),
        "n_mets": len(model.metabolites),
        "n_cobra_rxns": len(model.reactions),
        "n_cobra_genes": len(model.genes),
    }
    assign_standard_kcat(model, _uniprot([100.0]))
    remove_standard_kcat(model)

    assert list(model.ec.rxns) == snapshot["ec_rxns"]
    np.testing.assert_array_equal(model.ec.kcat, snapshot["ec_kcat"])
    assert list(model.ec.source) == snapshot["ec_source"]
    assert list(model.ec.enzymes) == snapshot["ec_enzymes"]
    assert list(model.ec.genes) == snapshot["ec_genes"]
    np.testing.assert_array_equal(model.ec.mw, snapshot["ec_mw"])
    assert len(model.metabolites) == snapshot["n_mets"]
    assert len(model.reactions) == snapshot["n_cobra_rxns"]
    assert len(model.genes) == snapshot["n_cobra_genes"]


def test_round_trip_with_fill_zero_kcat(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g2")],
        ec_rxns=["r1", "r2"],
        ec_kcats=[10.0, 0.0],
        ec_genes=["g1", "g2"], ec_enzymes=["g1", "g2"], ec_mws=[100.0, 100.0],
        rxn_to_ec_genes=[[0], [1]],
    )
    snap_kcat = model.ec.kcat.copy()
    snap_source = list(model.ec.source)

    assign_standard_kcat(model, _uniprot([100.0]), fill_zero_kcat=True)
    remove_standard_kcat(model)

    np.testing.assert_array_equal(model.ec.kcat, snap_kcat)
    assert list(model.ec.source) == snap_source


# --------------------------------------------------------------------------- #
# gecko_light
# --------------------------------------------------------------------------- #

def test_gecko_light_round_trip(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[], gecko_light=True, add_prot_pool=False,
    )
    snap_n_mets = len(model.metabolites)
    snap_n_rxns = len(model.reactions)

    assign_standard_kcat(model, _uniprot([100.0]))
    remove_standard_kcat(model)

    # In light mode, no topology was added or removed.
    assert len(model.metabolites) == snap_n_mets
    assert len(model.reactions) == snap_n_rxns
    # The standard enzyme is gone from ec, and the missing-rxn entry
    # (with 001_ prefix) is gone too.
    assert "standard" not in model.ec.enzymes
    assert not any(rid.startswith("001_") for rid in model.ec.rxns)
