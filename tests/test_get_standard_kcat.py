"""Tests for assign_standard_kcat."""
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.databases import UniprotDB
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import assign_standard_kcat


# --------------------------------------------------------------------------- #
# Fixture builders
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
    """Build an EcModel with metabolic reactions, a `prot_pool` met
    (for full mode), and a populated ec.

    `rxn_specs` items: (rxn_id, [(met_id, coeff, met_name, compartment), ...], gpr).
    """
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
    if add_prot_pool and not gecko_light:
        if "prot_pool" not in mets:
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
# Pre-flight
# --------------------------------------------------------------------------- #

def test_no_adapter_raises(tmp_path):
    model = _build_model_with_pool(
        _adapter(tmp_path),
        [("r1", [("A", -1.0, "alpha", "c")], "g1")],
        ec_rxns=[], ec_kcats=[],
    )
    model.adapter = None
    with pytest.raises(ValueError, match="adapter"):
        assign_standard_kcat(model, _uniprot([100.0]))


# --------------------------------------------------------------------------- #
# Standard MW computation
# --------------------------------------------------------------------------- #

def test_standard_mw_is_median_of_uniprot_mw(tmp_path):
    adapter = _adapter(tmp_path)
    # One reaction without GPR so it gets the standard pseudoenzyme.
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    db = _uniprot([10.0, 20.0, 30.0, 100.0, 1000.0])
    assign_standard_kcat(model, db)
    # MW for the standard pseudoenzyme is the median of the UniProt MWs.
    std_idx = model.ec.enzymes.index("standard")
    assert model.ec.mw[std_idx] == pytest.approx(30.0)


def test_standard_mw_handles_nan_in_uniprot(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    db = _uniprot([10.0, np.nan, 30.0, np.nan, 50.0])
    assign_standard_kcat(model, db)
    std_idx = model.ec.enzymes.index("standard")
    assert model.ec.mw[std_idx] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# Standard kcat computation
# --------------------------------------------------------------------------- #

def test_standard_kcat_is_median_of_nonzero_existing_kcats(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None),
         ("r2", [("A", -1.0, "alpha", "c")], "g1")],
        ec_rxns=["r2"],
        ec_kcats=[5.0],
        ec_genes=["g1"],
        ec_enzymes=["g1"],
        ec_mws=[100.0],
        rxn_to_ec_genes=[[0]],
    )
    # Add other ec entries with various kcats.
    model.ec.rxns = ["r2", "rA", "rB", "rC"]
    model.ec.kcat = np.array([5.0, 10.0, 20.0, 0.0])
    model.ec.source = ["", "", "", ""]
    model.ec.notes = [""] * 4
    model.ec.eccodes = [""] * 4
    model.ec.rxn_enz_mat = sparse.vstack([
        model.ec.rxn_enz_mat,
        sparse.csr_matrix((3, 1), dtype=float),
    ], format="csr")
    assign_standard_kcat(model, _uniprot([100.0]))
    # standard_kcat = median(5.0, 10.0, 20.0) = 10.0; rxn r1 (no GPR) gets that.
    r1_idx = model.ec.rxns.index("r1")
    assert model.ec.kcat[r1_idx] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Subsystem-specific kcat
# --------------------------------------------------------------------------- #

def test_subsystem_median_used_above_threshold(tmp_path):
    """A subsystem with >= threshold reactions uses the subsystem median."""
    adapter = _adapter(tmp_path)
    rxn_specs = []
    for i in range(15):
        rxn_specs.append((
            f"sub_r{i}",
            [(f"m{i}_in", -1.0, f"m{i}_in", "c"),
             (f"m{i}_out", 1.0, f"m{i}_out", "c")],
            f"g{i}",
        ))
    rxn_specs.append((
        "missing_r", [("X_in", -1.0, "X_in", "c"), ("X_out", 1.0, "X_out", "c")],
        None,  # no GPR; will get standard
    ))
    model = _build_model_with_pool(
        adapter, rxn_specs,
        ec_rxns=[f"sub_r{i}" for i in range(15)],
        ec_kcats=[2.0] * 15,  # all the same -> mean = 2.0
        ec_genes=[f"g{i}" for i in range(15)],
        ec_enzymes=[f"g{i}" for i in range(15)],
        ec_mws=[100.0] * 15,
        rxn_to_ec_genes=[[i] for i in range(15)],
    )
    # Set all 15 sub_r to subsystem "S" and missing_r also to "S".
    for r in model.reactions:
        if r.id.startswith("sub_r") or r.id == "missing_r":
            r.subsystem = "S"

    assign_standard_kcat(model, _uniprot([100.0]), threshold=10)
    # missing_r should get the subsystem median = 2.0, NOT the global
    # median of 2.0. (Same value here, but the path is "subsystem median".)
    missing_idx = model.ec.rxns.index("missing_r")
    assert model.ec.kcat[missing_idx] == pytest.approx(2.0)


def test_subsystem_value_is_median_not_mean(tmp_path):
    """Subsystem kcat is the median of its real kcats, not the mean:
    [1, 1, 100] has mean 34 but median 1."""
    adapter = _adapter(tmp_path)
    rxn_specs = [
        ("sub_r0", [("M0a", -1.0, "M0a", "c"), ("M0b", 1.0, "M0b", "c")], "g0"),
        ("sub_r1", [("M1a", -1.0, "M1a", "c"), ("M1b", 1.0, "M1b", "c")], "g1"),
        ("sub_r2", [("M2a", -1.0, "M2a", "c"), ("M2b", 1.0, "M2b", "c")], "g2"),
        ("missing_r", [("Xa", -1.0, "Xa", "c"), ("Xb", 1.0, "Xb", "c")], None),
    ]
    model = _build_model_with_pool(
        adapter, rxn_specs,
        ec_rxns=["sub_r0", "sub_r1", "sub_r2"],
        ec_kcats=[1.0, 1.0, 100.0],
        ec_genes=["g0", "g1", "g2"],
        ec_enzymes=["g0", "g1", "g2"],
        ec_mws=[100.0, 100.0, 100.0],
        rxn_to_ec_genes=[[0], [1], [2]],
    )
    for rid in ("sub_r0", "sub_r1", "sub_r2", "missing_r"):
        model.reactions.get_by_id(rid).subsystem = "S"

    assign_standard_kcat(model, _uniprot([100.0]), threshold=3)
    missing_idx = model.ec.rxns.index("missing_r")
    assert model.ec.kcat[missing_idx] == pytest.approx(1.0)


def test_subsystem_below_threshold_falls_back_to_standard(tmp_path):
    """Subsystem with < threshold reactions falls back to the global median."""
    adapter = _adapter(tmp_path)
    # 3 reactions in subsystem "S", threshold = 10 -> fallback.
    # Plus 5 reactions in another subsystem to influence the global median.
    rxn_specs = [
        ("sub_r0", [("M0a", -1.0, "M0a", "c"), ("M0b", 1.0, "M0b", "c")], "g0"),
        ("sub_r1", [("M1a", -1.0, "M1a", "c"), ("M1b", 1.0, "M1b", "c")], "g1"),
        ("sub_r2", [("M2a", -1.0, "M2a", "c"), ("M2b", 1.0, "M2b", "c")], "g2"),
        ("missing_r", [("Xa", -1.0, "Xa", "c"), ("Xb", 1.0, "Xb", "c")], None),
    ]
    model = _build_model_with_pool(
        adapter, rxn_specs,
        ec_rxns=["sub_r0", "sub_r1", "sub_r2"],
        ec_kcats=[100.0, 100.0, 100.0],
        ec_genes=["g0", "g1", "g2"],
        ec_enzymes=["g0", "g1", "g2"],
        ec_mws=[100.0, 100.0, 100.0],
        rxn_to_ec_genes=[[0], [1], [2]],
    )
    for rid in ("sub_r0", "sub_r1", "sub_r2"):
        model.reactions.get_by_id(rid).subsystem = "S"
    model.reactions.get_by_id("missing_r").subsystem = "S"

    assign_standard_kcat(model, _uniprot([100.0]), threshold=10)
    # Only 3 reactions in subsystem S, threshold=10 -> standardKcat = median(100, 100, 100) = 100.
    missing_idx = model.ec.rxns.index("missing_r")
    assert model.ec.kcat[missing_idx] == pytest.approx(100.0)


def test_subsystem_threshold_and_median_ignore_nan_kcats(tmp_path):
    """A NaN kcat sitting in a subsystem must not count toward that
    subsystem's threshold eligibility or poison its median: sub_r0 has a
    real kcat (100), sub_r1's kcat is still NaN (unset), and other_r (a
    different subsystem) has a real kcat (50) so the global standard kcat
    is a distinct value (75). With threshold=2, only sub_r0's kcat is
    real, so missing_r (no GPR, same subsystem) must fall back to the
    global standard kcat, not a NaN-poisoned subsystem median."""
    adapter = _adapter(tmp_path)
    rxn_specs = [
        ("sub_r0", [("M0a", -1.0, "M0a", "c"), ("M0b", 1.0, "M0b", "c")], "g0"),
        ("sub_r1", [("M1a", -1.0, "M1a", "c"), ("M1b", 1.0, "M1b", "c")], "g1"),
        ("other_r", [("M2a", -1.0, "M2a", "c"), ("M2b", 1.0, "M2b", "c")], "g2"),
        ("missing_r", [("Xa", -1.0, "Xa", "c"), ("Xb", 1.0, "Xb", "c")], None),
    ]
    model = _build_model_with_pool(
        adapter, rxn_specs,
        ec_rxns=["sub_r0", "sub_r1", "other_r"],
        ec_kcats=[100.0, np.nan, 50.0],
        ec_genes=["g0", "g1", "g2"],
        ec_enzymes=["g0", "g1", "g2"],
        ec_mws=[100.0, 100.0, 100.0],
        rxn_to_ec_genes=[[0], [1], [2]],
    )
    for rid in ("sub_r0", "sub_r1", "missing_r"):
        model.reactions.get_by_id(rid).subsystem = "S"

    assign_standard_kcat(model, _uniprot([100.0]), threshold=2)
    missing_idx = model.ec.rxns.index("missing_r")
    assert model.ec.kcat[missing_idx] == pytest.approx(75.0)


# --------------------------------------------------------------------------- #
# Topology additions
# --------------------------------------------------------------------------- #

def test_standard_pseudoenzyme_added_to_topology(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "prot_standard" in {m.id for m in model.metabolites}
    assert "usage_prot_standard" in {r.id for r in model.reactions}
    assert "standard" in model.ec.enzymes
    # Usage rxn: prot_pool -> prot_standard, bounds (0, 1000).
    usage = model.reactions.get_by_id("usage_prot_standard")
    stoich = {m.id: c for m, c in usage.metabolites.items()}
    assert stoich == {"prot_pool": -1.0, "prot_standard": 1.0}
    assert (usage.lower_bound, usage.upper_bound) == (0.0, 1000.0)


def test_standard_pseudoenzyme_skipped_for_gecko_light(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[], gecko_light=True, add_prot_pool=False,
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    # No metabolite or usage reaction added in light mode...
    assert "prot_standard" not in {m.id for m in model.metabolites}
    assert "usage_prot_standard" not in {r.id for r in model.reactions}
    # ... but ec.enzymes still extended.
    assert "standard" in model.ec.enzymes


def test_gecko_light_uses_4char_prefix_for_added_rxns(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[], gecko_light=True, add_prot_pool=False,
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    # gecko_light adds rxn IDs with "001_" prefix to ec.rxns.
    assert any(rid.startswith("001_") for rid in model.ec.rxns)


# --------------------------------------------------------------------------- #
# Reaction selection / filtering
# --------------------------------------------------------------------------- #

def test_reaction_without_gpr_gets_standard(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None),
            ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g1"),
        ],
        ec_rxns=["r2"], ec_kcats=[5.0],
        ec_genes=["g1"], ec_enzymes=["g1"], ec_mws=[100.0],
        rxn_to_ec_genes=[[0]],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "r1" in model.ec.rxns
    assert "r2" in model.ec.rxns


def test_reaction_with_gpr_but_no_ec_entry_gets_standard(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g_unmapped"),
        ],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "r1" in model.ec.rxns
    r1_idx = model.ec.rxns.index("r1")
    assert model.ec.source[r1_idx] == "standard"


def test_exchange_reaction_excluded(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("EX_alpha", [("A", -1.0, "alpha", "c")], None),  # boundary
            ("r2", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None),
        ],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "EX_alpha" not in model.ec.rxns
    assert "r2" in model.ec.rxns


def test_transport_reaction_excluded(tmp_path):
    """A reaction with the same metabolite name in two compartments is
    a transport, not a metabolic reaction."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("transport_r", [
                ("A_c", -1.0, "alpha", "c"),
                ("A_e", 1.0, "alpha", "e"),  # same name, diff compartment
            ], None),
            ("metabolic_r", [
                ("X", -1.0, "X_name", "c"),
                ("Y", 1.0, "Y_name", "c"),
            ], None),
        ],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "transport_r" not in model.ec.rxns
    assert "metabolic_r" in model.ec.rxns


def test_pseudoreaction_excluded_by_name(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("biomass_r", [
                ("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c"),
            ], None),
            ("normal_r", [
                ("X", -1.0, "X_name", "c"), ("Y", 1.0, "Y_name", "c"),
            ], None),
        ],
        ec_rxns=[], ec_kcats=[],
    )
    model.reactions.get_by_id("biomass_r").name = "growth pseudoreaction"
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "biomass_r" not in model.ec.rxns
    assert "normal_r" in model.ec.rxns


def test_custom_pseudo_rxns_tsv_excluded(tmp_path):
    adapter = _adapter(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pseudoRxns.tsv").write_text(
        "ignore_me\tShould be ignored\n", encoding="utf-8",
    )
    # Re-create adapter so it picks up the new path.
    adapter = ModelAdapter.from_folder(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [
            ("ignore_me", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None),
            ("keep_me", [("X", -1.0, "X_name", "c"), ("Y", 1.0, "Y_name", "c")], None),
        ],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    assert "ignore_me" not in model.ec.rxns
    assert "keep_me" in model.ec.rxns


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #

def test_idempotent_re_run_does_not_duplicate_standard(tmp_path):
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], None)],
        ec_rxns=[], ec_kcats=[],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    n_after_first = len(model.ec.rxns)
    assign_standard_kcat(model, _uniprot([100.0]))
    n_after_second = len(model.ec.rxns)
    assert n_after_first == n_after_second
    # Topology not duplicated either.
    n_standard_mets = sum(1 for m in model.metabolites if m.id == "prot_standard")
    assert n_standard_mets == 1


def test_idempotent_resets_source_on_real_enzyme_rxns_when_re_run(tmp_path):
    """A zero-kcat reaction tied to a real gene first receives the global
    standard kcat and source='standard'; once its source is reassigned
    (e.g. to 'manual'), a second run must leave its kcat and source
    untouched."""
    adapter = _adapter(tmp_path)
    # r2 has a real kcat so the global standard_kcat is computable;
    # r1 has zero kcat and will be filled with that standard.
    model = _build_model_with_pool(
        adapter,
        [
            ("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
            ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g2"),
        ],
        ec_rxns=["r1", "r2"],
        ec_kcats=[0.0, 7.0],
        ec_genes=["g1", "g2"], ec_enzymes=["g1", "g2"],
        ec_mws=[100.0, 100.0],
        rxn_to_ec_genes=[[0], [1]],
    )
    assign_standard_kcat(model, _uniprot([100.0]))
    # r1 was filled with standard (source='standard') but linked to real g1.
    r1_idx = model.ec.rxns.index("r1")
    assert model.ec.source[r1_idx] == "standard"
    assert model.ec.kcat[r1_idx] == pytest.approx(7.0)

    # Now add a real kcat for r1 (simulating the user fixing it manually).
    model.ec.kcat[r1_idx] = 50.0
    model.ec.source[r1_idx] = "manual"

    # Re-run: r1 should keep its real kcat (no longer unset).
    assign_standard_kcat(model, _uniprot([100.0]))
    r1_idx = model.ec.rxns.index("r1")
    assert model.ec.kcat[r1_idx] == 50.0
    assert model.ec.source[r1_idx] == "manual"


# --------------------------------------------------------------------------- #
# fill_zero_kcat
# --------------------------------------------------------------------------- #

def test_fill_zero_kcat_replaces_zeros_with_standard(tmp_path):
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
    assert model.ec.kcat[r2_idx] == pytest.approx(10.0)  # standard_kcat
    assert model.ec.source[r2_idx] == "standard"


def test_fill_zero_kcat_also_replaces_nan(tmp_path):
    """fill_zero_kcat treats an existing NaN kcat as unset, same as 0."""
    adapter = _adapter(tmp_path)
    model = _build_model_with_pool(
        adapter,
        [("r1", [("A", -1.0, "alpha", "c"), ("B", 1.0, "beta", "c")], "g1"),
         ("r2", [("B", -1.0, "beta", "c"), ("C", 1.0, "gamma", "c")], "g2")],
        ec_rxns=["r1", "r2"],
        ec_kcats=[10.0, np.nan],
        ec_genes=["g1", "g2"], ec_enzymes=["g1", "g2"], ec_mws=[100.0, 100.0],
        rxn_to_ec_genes=[[0], [1]],
    )
    assign_standard_kcat(model, _uniprot([100.0]), fill_zero_kcat=True)
    r2_idx = model.ec.rxns.index("r2")
    assert model.ec.kcat[r2_idx] == pytest.approx(10.0)
    assert model.ec.source[r2_idx] == "standard"


def test_fill_zero_kcat_false_leaves_zeros_alone(tmp_path):
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
    assign_standard_kcat(model, _uniprot([100.0]), fill_zero_kcat=False)
    r2_idx = model.ec.rxns.index("r2")
    assert model.ec.kcat[r2_idx] == 0.0




# --------------------------------------------------------------------------- #
# Subsystem names compare without regard to letter case
# --------------------------------------------------------------------------- #

def _model_with_subsystems(tmp_path, subsystems: dict[str, str], missing_subsystem):
    """12 reactions with a kcat (6 with 2.0 and 6 with 4.0) named in
    ``subsystems`` (by index), 5 reactions of subsystem ``Other`` with 100.0,
    and ``missing_r`` without genes in ``missing_subsystem``. The global
    standard kcat is 4.0."""
    adapter = _adapter(tmp_path)
    n = 17
    rxn_specs = [
        (f"r{i}", [(f"m{i}_in", -1.0, f"m{i}_in", "c"), (f"m{i}_out", 1.0, f"m{i}_out", "c")], f"g{i}")
        for i in range(n)
    ]
    rxn_specs.append(
        ("missing_r", [("X_in", -1.0, "X_in", "c"), ("X_out", 1.0, "X_out", "c")], None)
    )
    kcats = [2.0] * 6 + [4.0] * 6 + [100.0] * 5
    model = _build_model_with_pool(
        adapter, rxn_specs,
        ec_rxns=[f"r{i}" for i in range(n)],
        ec_kcats=kcats,
        ec_genes=[f"g{i}" for i in range(n)],
        ec_enzymes=[f"g{i}" for i in range(n)],
        ec_mws=[100.0] * n,
        rxn_to_ec_genes=[[i] for i in range(n)],
    )
    for i in range(n):
        model.reactions.get_by_id(f"r{i}").subsystem = (
            subsystems.get(i, "Other") if i < 12 else "Other"
        )
    model.reactions.get_by_id("missing_r").subsystem = missing_subsystem
    return model


def test_subsystem_names_differing_in_case_are_one_subsystem(tmp_path):
    """"Alpha" and "alpha" have 6 reactions each, below the threshold of 10 on
    their own; together they reach it and give their joint median, 3.0."""
    subsystems = {i: "Alpha" if i < 6 else "alpha" for i in range(12)}
    model = _model_with_subsystems(tmp_path, subsystems, "ALPHA")
    assign_standard_kcat(model, _uniprot([100.0]), threshold=10)
    assert model.ec.kcat[model.ec.rxns.index("missing_r")] == pytest.approx(3.0)


def test_empty_subsystem_gets_the_standard_kcat(tmp_path):
    subsystems = {i: "Alpha" for i in range(12)}
    model = _model_with_subsystems(tmp_path, subsystems, "")
    assign_standard_kcat(model, _uniprot([100.0]), threshold=10)
    assert model.ec.kcat[model.ec.rxns.index("missing_r")] == pytest.approx(4.0)


def test_empty_first_subsystem_of_a_list_gets_the_standard_kcat(tmp_path):
    subsystems = {i: "Alpha" for i in range(12)}
    model = _model_with_subsystems(tmp_path, subsystems, "")
    model.reactions.get_by_id("missing_r").subsystem = ["", "Alpha"]
    assign_standard_kcat(model, _uniprot([100.0]), threshold=10)
    assert model.ec.kcat[model.ec.rxns.index("missing_r")] == pytest.approx(4.0)
