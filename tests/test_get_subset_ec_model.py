"""Tests for get_subset_ec_model."""
import logging
from pathlib import Path

import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.ec_model.ec_data import EcData
from geckopy.utilities import get_subset_ec_model


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _adapter(tmp_path: Path) -> ModelAdapter:
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_big_ec_model(adapter: ModelAdapter) -> EcModel:
    """Big ec model with 3 catalysed reactions:

    R1 (gpr g1) catalysed by enzyme E1
    R2 (gpr g2) catalysed by enzyme E2
    R3 (gpr g3) catalysed by enzyme E3

    Plus the protein machinery for each:
        prot_E<N>, usage_prot_E<N>, prot_pool_exchange.
    """
    model = EcModel("big", adapter=adapter)

    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    D_c = cobra.Metabolite("D_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E1 = cobra.Metabolite("prot_E1", compartment="c")
    prot_E2 = cobra.Metabolite("prot_E2", compartment="c")
    prot_E3 = cobra.Metabolite("prot_E3", compartment="c")
    model.add_metabolites([
        A_c, B_c, C_c, D_c, pool, prot_E1, prot_E2, prot_E3,
    ])

    R1 = cobra.Reaction("R1")
    R1.add_metabolites({A_c: -1.0, prot_E1: -1/100, B_c: 1.0})
    R1.lower_bound = 0.0; R1.upper_bound = 1000.0
    R1.gene_reaction_rule = "g1"

    R2 = cobra.Reaction("R2")
    R2.add_metabolites({B_c: -1.0, prot_E2: -1/100, C_c: 1.0})
    R2.lower_bound = 0.0; R2.upper_bound = 1000.0
    R2.gene_reaction_rule = "g2"

    R3 = cobra.Reaction("R3")
    R3.add_metabolites({C_c: -1.0, prot_E3: -1/100, D_c: 1.0})
    R3.lower_bound = 0.0; R3.upper_bound = 1000.0
    R3.gene_reaction_rule = "g3"

    usage_E1 = cobra.Reaction("usage_prot_E1")
    usage_E1.add_metabolites({pool: -1.0, prot_E1: 1.0})
    usage_E1.lower_bound = 0.0; usage_E1.upper_bound = 1000.0

    usage_E2 = cobra.Reaction("usage_prot_E2")
    usage_E2.add_metabolites({pool: -1.0, prot_E2: 1.0})
    usage_E2.lower_bound = 0.0; usage_E2.upper_bound = 1000.0

    usage_E3 = cobra.Reaction("usage_prot_E3")
    usage_E3.add_metabolites({pool: -1.0, prot_E3: 1.0})
    usage_E3.lower_bound = 0.0; usage_E3.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0; pool_ex.upper_bound = 1000.0

    model.add_reactions([R1, R2, R3, usage_E1, usage_E2, usage_E3, pool_ex])

    n, g = 3, 3
    mat = sparse.lil_matrix((n, g), dtype=float)
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    mat[2, 2] = 1.0
    model.ec = EcData(
        rxns=["R1", "R2", "R3"],
        kcat=np.array([1.0, 2.0, 3.0]),
        source=["initial", "initial", "initial"],
        notes=["", "", ""],
        eccodes=["1.1.1.1", "2.2.2.2", "3.3.3.3"],
        genes=["g1", "g2", "g3"],
        enzymes=["E1", "E2", "E3"],
        mw=np.array([100.0, 200.0, 300.0]),
        sequence=["A", "B", "C"],
        concs=np.array([np.nan, np.nan, np.nan]),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def _build_small_gem(rxn_ids: list[str]) -> cobra.Model:
    """Conventional (non-ec) model with the named reactions and their
    associated genes. Stoichiometry is irrelevant for the subset
    operation."""
    model = cobra.Model("small")
    for rid in rxn_ids:
        rxn = cobra.Reaction(rid)
        rxn.gene_reaction_rule = f"g{rid[-1]}"
        model.add_reactions([rxn])
    return model


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_unknown_rxn_in_small_gem_raises(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1"])
    extra = cobra.Reaction("R_unknown")
    extra.gene_reaction_rule = "g_unknown"
    small.add_reactions([extra])
    with pytest.raises(ValueError, match="not found"):
        get_subset_ec_model(big, small)


# --------------------------------------------------------------------------- #
# No-op case
# --------------------------------------------------------------------------- #

def test_no_op_when_small_gem_matches_big(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2", "R3"])
    result = get_subset_ec_model(big, small)
    assert sorted(r.id for r in result.reactions) == sorted(
        r.id for r in big.reactions
    )
    assert sorted(g.id for g in result.genes) == sorted(
        g.id for g in big.genes
    )
    assert result.ec.rxns == big.ec.rxns
    assert result.ec.genes == big.ec.genes


# --------------------------------------------------------------------------- #
# Core subset trim
# --------------------------------------------------------------------------- #

def test_drops_rxn_and_gene_not_in_small_gem(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])  # drop R3/g3/E3
    result = get_subset_ec_model(big, small)

    rxn_ids = {r.id for r in result.reactions}
    gene_ids = {g.id for g in result.genes}

    assert "R3" not in rxn_ids
    assert "g3" not in gene_ids
    assert {"R1", "R2"}.issubset(rxn_ids)
    assert {"g1", "g2"}.issubset(gene_ids)


def test_kept_rxn_machinery_preserved(tmp_path):
    """Kept genes' usage_prot_*, prot_*, and pool exchange remain."""
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)

    rxn_ids = {r.id for r in result.reactions}
    met_ids = {m.id for m in result.metabolites}

    assert "usage_prot_E1" in rxn_ids
    assert "usage_prot_E2" in rxn_ids
    assert "prot_pool_exchange" in rxn_ids
    assert "prot_E1" in met_ids
    assert "prot_E2" in met_ids
    assert "prot_pool" in met_ids


def test_dropped_gene_machinery_removed(tmp_path):
    """When a gene is dropped, its catalysed rxn, usage_prot_<X> reaction
    and prot_<X> metabolite are all removed (no orphaned constraints)."""
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)

    rxn_ids = {r.id for r in result.reactions}
    met_ids = {m.id for m in result.metabolites}

    assert "R3" not in rxn_ids
    assert "usage_prot_E3" not in rxn_ids
    assert "prot_E3" not in met_ids
    # Kept-enzyme machinery is untouched.
    assert "usage_prot_E1" in rxn_ids
    assert "prot_E1" in met_ids


def test_ec_per_enzyme_fields_trimmed(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)

    assert result.ec.genes == ["g1", "g2"]
    assert result.ec.enzymes == ["E1", "E2"]
    np.testing.assert_array_equal(result.ec.mw, [100.0, 200.0])
    assert result.ec.sequence == ["A", "B"]
    assert result.ec.concs.shape == (2,)


def test_ec_per_rxn_fields_trimmed(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)

    assert result.ec.rxns == ["R1", "R2"]
    np.testing.assert_array_equal(result.ec.kcat, [1.0, 2.0])
    assert result.ec.source == ["initial", "initial"]
    assert result.ec.notes == ["", ""]
    assert result.ec.eccodes == ["1.1.1.1", "2.2.2.2"]


def test_rxn_enz_mat_shape_after_trim(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)

    assert result.ec.rxn_enz_mat.shape == (2, 2)
    # Identity-ish: R1 -> E1 (1.0), R2 -> E2 (1.0)
    dense = result.ec.rxn_enz_mat.toarray()
    np.testing.assert_array_equal(dense, np.eye(2))


# --------------------------------------------------------------------------- #
# Input not mutated
# --------------------------------------------------------------------------- #

def test_big_ec_model_not_mutated(tmp_path):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])

    big_rxn_ids_before = sorted(r.id for r in big.reactions)
    big_gene_ids_before = sorted(g.id for g in big.genes)
    big_ec_rxns_before = list(big.ec.rxns)
    big_ec_genes_before = list(big.ec.genes)
    big_ec_mat_shape_before = big.ec.rxn_enz_mat.shape

    _ = get_subset_ec_model(big, small)

    assert sorted(r.id for r in big.reactions) == big_rxn_ids_before
    assert sorted(g.id for g in big.genes) == big_gene_ids_before
    assert big.ec.rxns == big_ec_rxns_before
    assert big.ec.genes == big_ec_genes_before
    assert big.ec.rxn_enz_mat.shape == big_ec_mat_shape_before


# --------------------------------------------------------------------------- #
# Standard pseudo-gene preserved
# --------------------------------------------------------------------------- #

def test_standard_pseudo_gene_preserved(tmp_path):
    """If big has the `standard` pseudo-gene catalysing a rxn that's
    also in small_gem, both the gene and the rxn must survive."""
    big = _build_big_ec_model(_adapter(tmp_path))
    # Add a 4th rxn catalysed by the `standard` pseudo-gene.
    pool = big.metabolites.prot_pool
    prot_std = cobra.Metabolite("prot_standard", compartment="c")
    A_c = big.metabolites.A_c
    B_c = big.metabolites.B_c
    big.add_metabolites([prot_std])

    R_std = cobra.Reaction("R_std")
    R_std.add_metabolites({A_c: -1.0, prot_std: -1/100, B_c: 1.0})
    R_std.lower_bound = 0.0; R_std.upper_bound = 1000.0
    R_std.gene_reaction_rule = "standard"

    usage_std = cobra.Reaction("usage_prot_standard")
    usage_std.add_metabolites({pool: -1.0, prot_std: 1.0})
    usage_std.lower_bound = 0.0; usage_std.upper_bound = 1000.0

    big.add_reactions([R_std, usage_std])

    new_mat = sparse.lil_matrix(
        (big.ec.rxn_enz_mat.shape[0] + 1, big.ec.rxn_enz_mat.shape[1] + 1),
        dtype=float,
    )
    old_dense = big.ec.rxn_enz_mat.toarray()
    new_mat[: old_dense.shape[0], : old_dense.shape[1]] = old_dense
    new_mat[-1, -1] = 1.0
    big.ec = EcData(
        rxns=big.ec.rxns + ["R_std"],
        kcat=np.append(big.ec.kcat, 4.0),
        source=big.ec.source + ["initial"],
        notes=big.ec.notes + [""],
        eccodes=big.ec.eccodes + [""],
        genes=big.ec.genes + ["standard"],
        enzymes=big.ec.enzymes + ["standard"],
        mw=np.append(big.ec.mw, 400.0),
        sequence=big.ec.sequence + [""],
        concs=np.append(big.ec.concs, np.nan),
        rxn_enz_mat=new_mat.tocsr(),
    )

    # small_gem includes R_std (with `standard` gene assignment).
    small = _build_small_gem(["R1", "R_std"])
    # _build_small_gem inferred gene "g_d" from R_std's last char; fix.
    small.reactions.get_by_id("R_std").gene_reaction_rule = "standard"

    result = get_subset_ec_model(big, small)
    rxn_ids = {r.id for r in result.reactions}
    gene_ids = {g.id for g in result.genes}

    assert "R_std" in rxn_ids
    assert "standard" in gene_ids
    assert "standard" in result.ec.genes
    assert "R_std" in result.ec.rxns


# --------------------------------------------------------------------------- #
# REV / EXP suffix matching
# --------------------------------------------------------------------------- #

def test_rev_suffix_matched_to_canonical_id(tmp_path):
    """A reaction R2_REV in big should be kept iff R2 is in small_gem."""
    big = _build_big_ec_model(_adapter(tmp_path))
    # Replace R2's bounds and add an R2_REV variant pointing to E2.
    B_c = big.metabolites.B_c
    C_c = big.metabolites.C_c
    prot_E2 = big.metabolites.prot_E2
    R2_REV = cobra.Reaction("R2_REV")
    R2_REV.add_metabolites({C_c: -1.0, prot_E2: -1/100, B_c: 1.0})
    R2_REV.lower_bound = 0.0; R2_REV.upper_bound = 1000.0
    R2_REV.gene_reaction_rule = "g2"
    big.add_reactions([R2_REV])

    big.ec = EcData(
        rxns=big.ec.rxns + ["R2_REV"],
        kcat=np.append(big.ec.kcat, 2.0),
        source=big.ec.source + ["initial"],
        notes=big.ec.notes + [""],
        eccodes=big.ec.eccodes + ["2.2.2.2"],
        genes=big.ec.genes,
        enzymes=big.ec.enzymes,
        mw=big.ec.mw,
        sequence=big.ec.sequence,
        concs=big.ec.concs,
        rxn_enz_mat=sparse.vstack([
            big.ec.rxn_enz_mat,
            sparse.csr_matrix([[0.0, 1.0, 0.0]]),
        ]).tocsr(),
    )

    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)
    rxn_ids = {r.id for r in result.reactions}
    assert "R2_REV" in rxn_ids
    assert "R2" in rxn_ids


def test_exp_suffix_matched_to_canonical_id(tmp_path):
    """R1_EXP_2 in big should be kept iff R1 is in small_gem."""
    big = _build_big_ec_model(_adapter(tmp_path))
    A_c = big.metabolites.A_c
    B_c = big.metabolites.B_c
    prot_E1 = big.metabolites.prot_E1
    R1_EXP_2 = cobra.Reaction("R1_EXP_2")
    R1_EXP_2.add_metabolites({A_c: -1.0, prot_E1: -1/100, B_c: 1.0})
    R1_EXP_2.lower_bound = 0.0; R1_EXP_2.upper_bound = 1000.0
    R1_EXP_2.gene_reaction_rule = "g1"
    big.add_reactions([R1_EXP_2])

    big.ec = EcData(
        rxns=big.ec.rxns + ["R1_EXP_2"],
        kcat=np.append(big.ec.kcat, 1.0),
        source=big.ec.source + ["initial"],
        notes=big.ec.notes + [""],
        eccodes=big.ec.eccodes + ["1.1.1.1"],
        genes=big.ec.genes,
        enzymes=big.ec.enzymes,
        mw=big.ec.mw,
        sequence=big.ec.sequence,
        concs=big.ec.concs,
        rxn_enz_mat=sparse.vstack([
            big.ec.rxn_enz_mat,
            sparse.csr_matrix([[1.0, 0.0, 0.0]]),
        ]).tocsr(),
    )

    small = _build_small_gem(["R1", "R2"])
    result = get_subset_ec_model(big, small)
    rxn_ids = {r.id for r in result.reactions}
    assert "R1_EXP_2" in rxn_ids
    assert "R1" in rxn_ids


# --------------------------------------------------------------------------- #
# Warning on context-dependent protein constraints
# --------------------------------------------------------------------------- #

def test_constrained_usage_rxn_emits_warning(tmp_path, caplog):
    big = _build_big_ec_model(_adapter(tmp_path))
    big.reactions.get_by_id("usage_prot_E1").upper_bound = 0.5
    small = _build_small_gem(["R1", "R2"])
    with caplog.at_level(logging.WARNING):
        get_subset_ec_model(big, small)
    assert any(
        "protein-concentration constraints" in rec.message
        for rec in caplog.records
    )


def test_default_usage_bounds_emit_no_warning(tmp_path, caplog):
    big = _build_big_ec_model(_adapter(tmp_path))
    small = _build_small_gem(["R1", "R2"])
    with caplog.at_level(logging.WARNING):
        get_subset_ec_model(big, small)
    assert not any(
        "protein-concentration constraints" in rec.message
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# Reaction with an isozyme rule that survives the removal of one of its genes
# --------------------------------------------------------------------------- #

def _build_light_big_ec_model(adapter: ModelAdapter) -> EcModel:
    """Light ecModel: R1 (rule g1) and R2 (rule ``g1 and g2 or g3``), whose
    ec rows carry the ``001_`` prefix of a gecko-light model."""
    model = EcModel("big_light", adapter=adapter, gecko_light=True)
    mets = {n: cobra.Metabolite(f"{n}_c", compartment="c") for n in "ABC"}
    pool = cobra.Metabolite("prot_pool", compartment="c")
    model.add_metabolites([*mets.values(), pool])

    r1 = cobra.Reaction("R1")
    r1.add_metabolites({mets["A"]: -1.0, mets["B"]: 1.0})
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("R2")
    r2.add_metabolites({mets["B"]: -1.0, mets["C"]: 1.0})
    r2.gene_reaction_rule = "g1 and g2 or g3"
    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.upper_bound = 1000.0
    model.add_reactions([r1, r2, pool_ex])

    mat = sparse.lil_matrix((2, 3), dtype=float)
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    mat[1, 2] = 1.0
    model.ec = EcData(
        rxns=["001_R1", "001_R2"],
        kcat=np.array([1.0, 2.0]),
        source=["initial", "initial"],
        notes=["", ""],
        eccodes=["1.1.1.1", "2.2.2.2"],
        genes=["g1", "g2", "g3"],
        enzymes=["E1", "E2", "E3"],
        mw=np.array([100.0, 200.0, 300.0]),
        sequence=["A", "B", "C"],
        concs=np.array([np.nan, np.nan, np.nan]),
        rxn_enz_mat=mat.tocsr(),
        gecko_light=True,
    )
    return model


def _small_gem_without_r2_and_g3() -> cobra.Model:
    """Context model with R1 and the genes g1 and g2, lacking R2 and g3."""
    small = cobra.Model("small")
    r1 = cobra.Reaction("R1")
    r1.gene_reaction_rule = "g1"
    small.add_reactions([r1])
    small.genes.append(cobra.Gene("g2"))
    return small


def test_rxn_with_surviving_rule_and_missing_in_small_gem(tmp_path):
    """R2 survives the removal of g3 (g1 and g2 satisfy its rule) and is then
    removed because the context model lacks it; the gene g3 is not removed twice."""
    big = _build_light_big_ec_model(_adapter(tmp_path))
    result = get_subset_ec_model(big, _small_gem_without_r2_and_g3())

    assert {r.id for r in result.reactions} == {"R1", "prot_pool_exchange"}
    assert {g.id for g in result.genes} == {"g1"}
    assert result.ec.genes == ["g1", "g2"]
    assert result.ec.enzymes == ["E1", "E2"]


def test_surviving_rxn_loses_removed_gene_from_rule(tmp_path):
    """A kept reaction no longer refers to a gene that left model.genes."""
    big = _build_light_big_ec_model(_adapter(tmp_path))
    small = _small_gem_without_r2_and_g3()
    small.add_reactions([cobra.Reaction("R2")])
    result = get_subset_ec_model(big, small)

    r2 = result.reactions.get_by_id("R2")
    assert {g.id for g in r2.genes} == {"g1", "g2"}
    assert "g3" not in r2.gene_reaction_rule
    assert {g.id for g in result.genes} == {"g1", "g2"}


def test_light_model_keeps_ec_rows_of_kept_reactions(tmp_path):
    """The ``001_`` prefix of the ec rows of a light model is not part of the
    reaction id the rows are matched against."""
    big = _build_light_big_ec_model(_adapter(tmp_path))
    result = get_subset_ec_model(big, _small_gem_without_r2_and_g3())

    assert result.ec.rxns == ["001_R1"]
    np.testing.assert_array_equal(result.ec.kcat, [1.0])
    assert result.ec.eccodes == ["1.1.1.1"]
    assert result.ec.rxn_enz_mat.shape == (1, 2)


def test_identity_subset_of_light_model_is_a_copy(tmp_path):
    big = _build_light_big_ec_model(_adapter(tmp_path))
    small = cobra.Model("small")
    for rid, rule in (("R1", "g1"), ("R2", "g1 and g2 or g3")):
        rxn = cobra.Reaction(rid)
        rxn.gene_reaction_rule = rule
        small.add_reactions([rxn])
    result = get_subset_ec_model(big, small)

    assert result is not big
    assert sorted(r.id for r in result.reactions) == sorted(r.id for r in big.reactions)
    assert result.ec.rxns == ["001_R1", "001_R2"]


def test_light_model_ec_rows_without_gene_removal(tmp_path):
    big = _build_light_big_ec_model(_adapter(tmp_path))
    small = cobra.Model("small")
    r1 = cobra.Reaction("R1")
    r1.gene_reaction_rule = "g1"
    small.add_reactions([r1])
    small.genes.extend([cobra.Gene("g2"), cobra.Gene("g3")])
    result = get_subset_ec_model(big, small)

    assert [r.id for r in result.reactions if r.id.startswith("R")] == ["R1"]
    assert result.ec.rxns == ["001_R1"]
    assert result.ec.rxn_enz_mat.shape == (1, 3)


# --------------------------------------------------------------------------- #
# Metabolites and genes that no reaction uses
# --------------------------------------------------------------------------- #

def test_unused_metabolites_and_genes_are_removed(tmp_path):
    """MATLAB's removeGenes and removeReactions drop every metabolite and gene
    without a reaction, also those that were unused in the big model."""
    big = _build_light_big_ec_model(_adapter(tmp_path))
    big.add_metabolites([cobra.Metabolite("X_c", compartment="c")])
    big.genes.append(cobra.Gene("standard"))
    small = cobra.Model("small")
    for rid, rule in (("R1", "g1"), ("R2", "g1 and g2 or g3")):
        rxn = cobra.Reaction(rid)
        rxn.gene_reaction_rule = rule
        small.add_reactions([rxn])
    result = get_subset_ec_model(big, small)

    assert not result.metabolites.has_id("X_c")
    assert not result.genes.has_id("standard")
    assert {g.id for g in result.genes} == {"g1", "g2", "g3"}
    assert all(m.reactions for m in result.metabolites)
    assert big.metabolites.has_id("X_c")
    assert big.genes.has_id("standard")
