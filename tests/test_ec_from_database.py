"""Tests for fill_eccodes_from_database."""
import logging
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from geckopy import EcModel, ModelAdapter
from geckopy.databases import UniprotDB
from geckopy.ec_model.ec_data import EcData
from geckopy.get_enzyme_data import fill_eccodes_from_database


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _minimal_adapter(tmp_path: Path) -> ModelAdapter:
    """Adapter with a real on-disk path so get_uniprot_ids_from_table works."""
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\n'
    )
    return ModelAdapter.from_folder(tmp_path)


def _build_ec_model(
    adapter: ModelAdapter,
    rxns: list[str],
    genes: list[str],
    rxn_to_genes: list[list[int]],
    *,
    initial_eccodes: list[str] | None = None,
) -> EcModel:
    """Build an EcModel with manually-populated ec.

    ``rxn_to_genes`` is a list of lists: ``rxn_to_genes[i]`` is the
    indices into ``genes`` for the proteins catalysing reaction
    ``rxns[i]``.
    """
    model = EcModel("test", adapter=adapter)
    n = len(rxns)
    g = len(genes)

    if initial_eccodes is None:
        initial_eccodes = [""] * n

    mat = sparse.lil_matrix((n, g), dtype=float)
    for i, gene_indices in enumerate(rxn_to_genes):
        for j in gene_indices:
            mat[i, j] = 1.0

    model.ec = EcData(
        rxns=list(rxns),
        kcat=np.full(n, np.nan, dtype=float),
        source=[""] * n,
        notes=[""] * n,
        eccodes=list(initial_eccodes),
        genes=list(genes),
        enzymes=[""] * g,
        mw=np.zeros(g, dtype=float),
        sequence=[""] * g,
        concs=np.full(g, np.nan, dtype=float),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def _uniprot(rows: list[tuple[str, str, str, float]]) -> UniprotDB:
    """Build a UniprotDB from ``(uniprot_id, gene_name, ec, mw)`` rows."""
    return UniprotDB(
        ids=[r[0] for r in rows],
        genes=[r[1] for r in rows],
        eccodes=[r[2] for r in rows],
        mw=np.array([r[3] for r in rows], dtype=float),
        sequences=[""] * len(rows),
    )


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_empty_ec_rxns_is_a_noop(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, [], [], [])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == []


def test_single_reaction_single_gene_one_protein(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_reaction_with_no_genes_gets_empty_string(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == [""]


def test_gene_not_in_db_yields_empty_string(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g_missing"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == [""]


# --------------------------------------------------------------------------- #
# Multi-gene complexes
# --------------------------------------------------------------------------- #

def test_multi_gene_complex_intersection(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter, ["r1"], ["g1", "g2"], [[0, 1]],
    )
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g2", "1.1.1.1", 100.0),  # same EC -> intersection
    ])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_multi_gene_complex_no_intersection_falls_back_to_union(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter, ["r1"], ["g1", "g2"], [[0, 1]],
    )
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g2", "2.2.2.2", 100.0),  # disjoint -> union
    ])
    fill_eccodes_from_database(model, db)
    assert set(model.ec.eccodes[0].split(";")) == {"1.1.1.1", "2.2.2.2"}


def test_multiple_reactions_each_resolved_independently(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter,
        ["r1", "r2", "r3"],
        ["g1", "g2", "g3"],
        [[0], [1], [2]],
    )
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g2", "2.2.2.2", 100.0),
        ("P3", "g3", "3.3.3.3", 100.0),
    ])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1", "2.2.2.2", "3.3.3.3"]


# --------------------------------------------------------------------------- #
# ec_rxns selector
# --------------------------------------------------------------------------- #

def test_ec_rxns_subset_only_updates_specified(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter,
        ["r1", "r2", "r3"],
        ["g1", "g2", "g3"],
        [[0], [1], [2]],
        initial_eccodes=["", "preexisting", ""],
    )
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g2", "2.2.2.2", 100.0),
        ("P3", "g3", "3.3.3.3", 100.0),
    ])
    fill_eccodes_from_database(model, db, ec_rxns=["r1", "r3"])
    assert model.ec.eccodes == ["1.1.1.1", "preexisting", "3.3.3.3"]


def test_ec_rxns_empty_iterable_is_noop(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter, ["r1"], ["g1"], [[0]],
        initial_eccodes=["preexisting"],
    )
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db, ec_rxns=[])
    assert model.ec.eccodes == ["preexisting"]


def test_ec_rxns_unknown_id_raises(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    with pytest.raises(ValueError, match="not present in model.ec.rxns"):
        fill_eccodes_from_database(model, db, ec_rxns=["nonexistent"])


def test_no_mask_overwrites_all_existing_eccodes(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter, ["r1", "r2"], ["g1"], [[0], [0]],
        initial_eccodes=["preexisting1", "preexisting2"],
    )
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1", "1.1.1.1"]


# --------------------------------------------------------------------------- #
# Action handling
# --------------------------------------------------------------------------- #

def test_action_display_emits_aggregated_warning_on_conflict(tmp_path, caplog):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    # `g1` matches two distinct DB entries with different EC codes.
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g1", "2.2.2.2", 100.0),
    ])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, db, action="display")
    # Aggregated message specific to fill_eccodes_from_database.
    assert "gene-protein conflict" in caplog.text
    assert "rxn 'r1'" in caplog.text
    assert "P1" in caplog.text
    assert "P2" in caplog.text


def test_action_ignore_does_not_emit_aggregated_warning(tmp_path, caplog):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g1", "2.2.2.2", 100.0),
    ])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, db, action="ignore")
    assert "gene-protein conflict" not in caplog.text


def test_action_invalid_raises(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    with pytest.raises(ValueError, match="action must be"):
        fill_eccodes_from_database(model, db, action="something_else")


def test_no_aggregated_warning_when_no_conflicts(tmp_path, caplog):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, db, action="display")
    assert "gene-protein conflict" not in caplog.text


# --------------------------------------------------------------------------- #
# Adapter wiring
# --------------------------------------------------------------------------- #

def test_no_adapter_raises(tmp_path):
    model = _build_ec_model(_minimal_adapter(tmp_path), ["r1"], ["g1"], [[0]])
    model.adapter = None
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    with pytest.raises(ValueError, match="No ModelAdapter available"):
        fill_eccodes_from_database(model, db)


def test_get_uniprot_compatible_genes_transformation_used(tmp_path):
    """A subclass that strips a prefix should let model genes match
    DB gene names without the prefix."""
    class StripPrefixAdapter(ModelAdapter):
        def get_uniprot_compatible_genes(self, in_genes):
            return [g.removeprefix("MODEL_") for g in in_genes]

    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\n'
    )
    adapter = StripPrefixAdapter.from_folder(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["MODEL_g1"], [[0]])
    db = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_uniprot_conversion_table_path(tmp_path):
    """When data/uniprotConversion.tsv exists, model genes are mapped
    directly to UniProt IDs and looked up by ID rather than gene name."""
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\n'
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "uniprotConversion.tsv").write_text(
        "model_gene\tuniprot_id\nmy_g1\tP_target\n"
    )
    adapter = ModelAdapter.from_folder(tmp_path)

    model = _build_ec_model(adapter, ["r1"], ["my_g1"], [[0]])
    # Note the DB gene name is empty so name-based lookup would fail;
    # only ID-based lookup via the conversion table can resolve.
    db = _uniprot([
        ("P_other", "", "9.9.9.9", 100.0),
        ("P_target", "", "1.1.1.1", 100.0),
    ])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_uniprot_conversion_table_ignores_unmapped_genes(tmp_path):
    """A gene absent from the conversion table is silently skipped."""
    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\norg_name = "test"\n'
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Only g1 is mapped; g2 is not.
    (data_dir / "uniprotConversion.tsv").write_text(
        "model_gene\tuniprot_id\ng1\tP1\n"
    )
    adapter = ModelAdapter.from_folder(tmp_path)

    model = _build_ec_model(adapter, ["r1"], ["g1", "g2"], [[0, 1]])
    db = _uniprot([("P1", "", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, db)
    # Only g1's EC contributes; g2 is not in the table and not in the DB by name.
    assert model.ec.eccodes == ["1.1.1.1"]


# --------------------------------------------------------------------------- #
# Multi-protein-per-gene (the main reason gene_to_protein_indices is a list)
# --------------------------------------------------------------------------- #

def test_gene_with_multiple_proteins_picks_lightest(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 200.0),
        ("P2", "g1", "1.1.1.1", 100.0),  # lighter, same EC
        ("P3", "g1", "1.1.1.1", 300.0),
    ])
    fill_eccodes_from_database(model, db)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_conflicts_aggregated_across_multiple_reactions(tmp_path, caplog):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(
        adapter,
        ["r1", "r2"],
        ["g1", "g2"],
        [[0], [1]],
    )
    db = _uniprot([
        ("P1", "g1", "1.1.1.1", 100.0),
        ("P2", "g1", "2.2.2.2", 100.0),
        ("P3", "g2", "3.3.3.3", 100.0),
        ("P4", "g2", "4.4.4.4", 100.0),
    ])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, db, action="display")
    assert "2 gene-protein conflict" in caplog.text
    assert "across 2 reaction" in caplog.text
    assert "rxn 'r1'" in caplog.text
    assert "rxn 'r2'" in caplog.text


# --------------------------------------------------------------------------- #
# KEGG fallback
# --------------------------------------------------------------------------- #

def _kegg(rows: list[tuple[str, str, str, str, float]]) -> "object":
    """Build a KeggDB from ``(uniprot_id, gene, kegg_gene, ec, mw)`` rows."""
    from geckopy.databases.kegg_loader import KeggDB
    return KeggDB(
        uniprot_ids=[r[0] for r in rows],
        genes=[r[1] for r in rows],
        kegg_genes=[r[2] for r in rows],
        eccodes=[r[3] for r in rows],
        mw=np.array([r[4] for r in rows], dtype=float),
        pathways=[""] * len(rows),
        sequences=[""] * len(rows),
    )


def test_kegg_fallback_when_uniprot_empty(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([])  # nothing in UniProt
    kegg = _kegg([("Q1", "g1", "K1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, uniprot, kegg_db=kegg)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_kegg_fallback_when_uniprot_returns_wildcard(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([("P1", "g1", "1.1.1.-", 100.0)])
    kegg = _kegg([("Q1", "g1", "K1", "1.1.1.42", 100.0)])
    fill_eccodes_from_database(model, uniprot, kegg_db=kegg)
    assert model.ec.eccodes == ["1.1.1.42"]


def test_kegg_not_used_when_uniprot_returns_complete_ec(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([("P1", "g1", "1.1.1.1", 100.0)])
    kegg = _kegg([("Q1", "g1", "K1", "9.9.9.9", 100.0)])
    fill_eccodes_from_database(model, uniprot, kegg_db=kegg)
    assert model.ec.eccodes == ["1.1.1.1"]


def test_kegg_fallback_with_no_match_leaves_empty(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([])
    kegg = _kegg([("Q1", "g_other", "K1", "1.1.1.1", 100.0)])
    fill_eccodes_from_database(model, uniprot, kegg_db=kegg)
    assert model.ec.eccodes == [""]


def test_kegg_fallback_tags_conflict_source(tmp_path, caplog):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([])
    kegg = _kegg([
        ("Q1", "g1", "K1", "1.1.1.1", 100.0),
        ("Q2", "g1", "K2", "2.2.2.2", 100.0),
    ])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, uniprot, kegg_db=kegg, action="display")
    assert "(kegg)" in caplog.text


def test_no_kegg_arg_preserves_original_behaviour(tmp_path):
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    uniprot = _uniprot([])
    fill_eccodes_from_database(model, uniprot)
    assert model.ec.eccodes == [""]


# --------------------------------------------------------------------------- #
# Gene cells with several names
# --------------------------------------------------------------------------- #

def test_gene_matches_second_name_of_a_gene_cell(tmp_path):
    from geckopy.databases import load_uniprot_tsv

    tsv = tmp_path / "uniprot.tsv"
    tsv.write_text(
        "Entry\tGene Names\tEC number\tMass\tSequence\n"
        "P1\tg0; g1\t1.1.1.1\t100\tMK\n"
    )
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    fill_eccodes_from_database(model, load_uniprot_tsv(tsv))
    assert model.ec.eccodes == ["1.1.1.1"]


def test_gene_in_two_entries_is_handled_as_with_separate_rows(tmp_path, caplog):
    from geckopy.databases import load_uniprot_tsv

    tsv = tmp_path / "uniprot.tsv"
    tsv.write_text(
        "Entry\tGene Names\tEC number\tMass\tSequence\n"
        "P1\tg0; g1\t1.1.1.1\t100\tMK\n"
        "P2\tg1\t2.2.2.2\t200\tMK\n"
    )
    adapter = _minimal_adapter(tmp_path)
    model = _build_ec_model(adapter, ["r1"], ["g1"], [[0]])
    with caplog.at_level(logging.WARNING):
        fill_eccodes_from_database(model, load_uniprot_tsv(tsv))
    assert model.ec.eccodes == ["1.1.1.1"]
    assert "g1" in caplog.text
