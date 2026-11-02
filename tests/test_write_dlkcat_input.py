"""Tests for write_dlkcat_input."""
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geckopy.databases import DLKcatIgnoreLists
from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.gather_kcats import write_dlkcat_input


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _ec_model(
    rxn_specs: list[tuple[str, list[tuple[str, float, str]]]],
    *,
    gecko_light: bool = False,
    ec_rxn_prefix: str = "",
    genes_per_rxn: dict[str, list[str]] | None = None,
    gene_sequences: dict[str, str] | None = None,
    met_smiles: dict[str, str] | None = None,
) -> EcModel:
    """Build an EcModel with reactions, genes, sequences, and SMILES.

    rxn_specs: list of (rxn_id, [(met_id, coeff, met_name), ...]).
    gecko_light: build the model in gecko-light layout when True.
    ec_rxn_prefix: prefix prepended to each rxn_id when building
        model.ec.rxns (e.g. the isozyme-index prefix used in gecko-light).
    genes_per_rxn: {rxn_id: [gene_id, ...]}.
    gene_sequences: {gene_id: sequence}.
    met_smiles: {met_id: smiles}.
    """
    genes_per_rxn = genes_per_rxn or {}
    gene_sequences = gene_sequences or {}
    met_smiles = met_smiles or {}

    model = EcModel("test", gecko_light=gecko_light)

    mets: dict[str, cobra.Metabolite] = {}
    for _, met_list in rxn_specs:
        for met_id, _, met_name in met_list:
            if met_id not in mets:
                m = cobra.Metabolite(met_id, compartment="c")
                m.name = met_name
                if met_id in met_smiles:
                    m.annotation["smiles"] = met_smiles[met_id]
                mets[met_id] = m
    model.add_metabolites(list(mets.values()))

    for rxn_id, met_list in rxn_specs:
        rxn = cobra.Reaction(rxn_id)
        rxn.lower_bound = 0.0
        rxn.upper_bound = 1000.0
        rxn.add_metabolites({mets[mid]: c for mid, c, _ in met_list})
        model.add_reactions([rxn])

    # Build ec.genes from the union of all genes_per_rxn values.
    all_genes: list[str] = []
    seen = set()
    for rxn_id, _ in rxn_specs:
        for g in genes_per_rxn.get(rxn_id, []):
            if g not in seen:
                seen.add(g)
                all_genes.append(g)

    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    sequences = [gene_sequences.get(g, "") for g in all_genes]

    n = len(rxn_specs)
    g = len(all_genes)
    mat = sparse.lil_matrix((n, g), dtype=float)
    for i, (rxn_id, _) in enumerate(rxn_specs):
        for gname in genes_per_rxn.get(rxn_id, []):
            mat[i, gene_to_idx[gname]] = 1.0

    model.ec = EcData(
        gecko_light=gecko_light,
        rxns=[ec_rxn_prefix + r for r, _ in rxn_specs],
        kcat=np.full(n, np.nan, dtype=float),
        source=[""] * n,
        notes=[""] * n,
        eccodes=[""] * n,
        genes=list(all_genes),
        enzymes=list(all_genes),
        mw=np.zeros(g, dtype=float),
        sequence=sequences,
        concs=np.full(g, np.nan, dtype=float),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def _ignore_lists(
    *,
    ignore_names: list[str] | None = None,
    ignore_smiles: list[str] | None = None,
    currency_pairs: list[tuple[str, str]] | None = None,
) -> DLKcatIgnoreLists:
    return DLKcatIgnoreLists(
        ignore_names=ignore_names or [],
        ignore_smiles=ignore_smiles or [],
        currency_pairs=currency_pairs or [],
    )


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read the written TSV (no header) into a DataFrame."""
    if path.stat().st_size == 0:
        return pd.DataFrame(
            columns=["rxn_id", "gene", "substrate", "smiles", "sequence", "kcat"]
        )
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["rxn_id", "gene", "substrate", "smiles", "sequence", "kcat"],
        keep_default_na=False,
    )


# --------------------------------------------------------------------------- #
# File-handling
# --------------------------------------------------------------------------- #

def test_existing_file_refused_without_overwrite(tmp_path):
    out = tmp_path / "DLKcat.tsv"
    out.write_text("preexisting", encoding="utf-8")
    model = _ec_model([])
    with pytest.raises(FileExistsError):
        write_dlkcat_input(model, out, _ignore_lists())


def test_existing_file_overwritten_when_requested(tmp_path):
    out = tmp_path / "DLKcat.tsv"
    out.write_text("preexisting", encoding="utf-8")
    model = _ec_model([])
    write_dlkcat_input(model, out, _ignore_lists(), overwrite=True)
    # File was replaced; no rows since model is empty.
    assert out.read_text(encoding="utf-8") == ""


def test_empty_model_writes_empty_file(tmp_path):
    model = _ec_model([])
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    assert df.empty
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------- #
# Basic flow
# --------------------------------------------------------------------------- #

def test_single_reaction_single_substrate_single_gene(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", 1.0, "beta")])],
        genes_per_rxn={"r1": ["g1"]},
        gene_sequences={"g1": "MASEQ"},
        met_smiles={"A": "C(C)O"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["rxn_id"] == "r1"
    assert row["gene"] == "g1"
    assert row["substrate"] == "alpha"
    assert row["smiles"] == "C(C)O"
    assert row["sequence"] == "MASEQ"
    assert row["kcat"] == "NA"


def test_complex_with_multiple_genes_yields_multiple_rows(tmp_path):
    """A reaction with two subunits and two substrates yields 4 rows."""
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", -1.0, "beta"), ("C", 1.0, "gamma")])],
        genes_per_rxn={"r1": ["g1", "g2"]},
        gene_sequences={"g1": "M1", "g2": "M2"},
        met_smiles={"A": "SMI_A", "B": "SMI_B"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    assert len(df) == 4
    triples = sorted(zip(df["substrate"], df["gene"]))
    assert triples == [("alpha", "g1"), ("alpha", "g2"),
                       ("beta", "g1"), ("beta", "g2")]


def test_only_substrates_not_products_emitted(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", 1.0, "beta")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A", "B": "SMI_B"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    assert list(df["substrate"]) == ["alpha"]


# --------------------------------------------------------------------------- #
# Ignore lists
# --------------------------------------------------------------------------- #

def test_ignored_metabolite_by_name_dropped(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "Alpha"), ("B", -1.0, "H2O")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A", "B": "O"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(ignore_names=["h2o"]),
    )
    assert list(df["substrate"]) == ["Alpha"]


def test_ignored_metabolite_by_smiles_dropped(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", -1.0, "Water")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A", "B": "O"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(ignore_smiles=["O"]),
    )
    assert list(df["substrate"]) == ["alpha"]


def test_protein_pseudometabolites_dropped(tmp_path):
    """Metabolites with IDs starting with `prot_` are protein-usage
    pseudometabolites and must be dropped."""
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("prot_pool_g1", -1.0, "prot_pool_g1")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    assert list(df["substrate"]) == ["alpha"]


# --------------------------------------------------------------------------- #
# Currency pair handling
# --------------------------------------------------------------------------- #

def test_currency_pair_removed_when_other_substrate_remains(tmp_path):
    """ATP + glucose -> ADP + glucose-6-P. With currency pair (ATP, ADP),
    both should be stripped, leaving glucose as the only substrate row."""
    model = _ec_model(
        [("r1", [
            ("atp", -1.0, "ATP"),
            ("glc", -1.0, "glucose"),
            ("adp", 1.0, "ADP"),
            ("g6p", 1.0, "glucose-6-phosphate"),
        ])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"atp": "SMI_ATP", "glc": "SMI_GLC"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(currency_pairs=[("atp", "adp")]),
    )
    assert list(df["substrate"]) == ["glucose"]


def test_currency_pair_preserved_when_no_other_substrate(tmp_path):
    """Pure ATP -> ADP reaction (no other substrates). Currency pair
    should NOT be stripped to avoid leaving the reaction empty."""
    model = _ec_model(
        [("r1", [
            ("atp", -1.0, "ATP"),
            ("adp", 1.0, "ADP"),
        ])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"atp": "SMI_ATP"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(currency_pairs=[("atp", "adp")]),
    )
    # ATP is preserved as the substrate.
    assert list(df["substrate"]) == ["ATP"]


def test_currency_pair_only_when_both_present(tmp_path):
    """A reaction with only ATP (no ADP) is unaffected by the currency
    pair rule."""
    model = _ec_model(
        [("r1", [
            ("atp", -1.0, "ATP"),
            ("X", 1.0, "X"),
        ])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"atp": "SMI_ATP"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(currency_pairs=[("atp", "adp")]),
    )
    assert list(df["substrate"]) == ["ATP"]


# --------------------------------------------------------------------------- #
# only_with_smiles
# --------------------------------------------------------------------------- #

def test_only_with_smiles_drops_no_smiles_rows(tmp_path):
    model = _ec_model(
        [
            ("r1", [("A", -1.0, "alpha")]),
            ("r2", [("B", -1.0, "beta")]),
        ],
        genes_per_rxn={"r1": ["g1"], "r2": ["g2"]},
        met_smiles={"A": "SMI_A"},  # B has no SMILES
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists(), only_with_smiles=True)
    assert list(df["substrate"]) == ["alpha"]


def test_without_only_with_smiles_writes_None_placeholder(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha")])],
        genes_per_rxn={"r1": ["g1"]},
        # No SMILES.
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists(), only_with_smiles=False)
    assert list(df["smiles"]) == ["None"]


# --------------------------------------------------------------------------- #
# ec_rxns selector
# --------------------------------------------------------------------------- #

def test_ec_rxns_subset_only_emitted(tmp_path):
    model = _ec_model(
        [
            ("r1", [("A", -1.0, "alpha")]),
            ("r2", [("B", -1.0, "beta")]),
        ],
        genes_per_rxn={"r1": ["g1"], "r2": ["g2"]},
        met_smiles={"A": "SMI_A", "B": "SMI_B"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(
        model, out, _ignore_lists(), ec_rxns=["r1"],
    )
    assert list(df["rxn_id"]) == ["r1"]


def test_ec_rxns_unknown_id_raises(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A"},
    )
    out = tmp_path / "DLKcat.tsv"
    with pytest.raises(ValueError, match="not present in model.ec.rxns"):
        write_dlkcat_input(
            model, out, _ignore_lists(), ec_rxns=["nonexistent"],
        )


# --------------------------------------------------------------------------- #
# gecko_light prefix
# --------------------------------------------------------------------------- #

def test_gecko_light_strips_4char_prefix_to_find_reaction(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha")])],
        gecko_light=True,
        ec_rxn_prefix="001_",
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    # ec.rxns = ["001_r1"]; cobra reactions = ["r1"]. ec.rxns ID kept in output.
    assert list(df["rxn_id"]) == ["001_r1"]


# --------------------------------------------------------------------------- #
# On-disk file format matches DataFrame
# --------------------------------------------------------------------------- #

def test_written_file_matches_returned_dataframe(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha")])],
        genes_per_rxn={"r1": ["g1"]},
        gene_sequences={"g1": "MAB"},
        met_smiles={"A": "SMI_A"},
    )
    out = tmp_path / "DLKcat.tsv"
    df = write_dlkcat_input(model, out, _ignore_lists())
    on_disk = _read_tsv(out)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), on_disk.reset_index(drop=True),
        check_dtype=False,
    )


def test_written_file_has_no_header(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha")])],
        genes_per_rxn={"r1": ["g1"]},
        met_smiles={"A": "SMI_A"},
    )
    out = tmp_path / "DLKcat.tsv"
    write_dlkcat_input(model, out, _ignore_lists())
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    # Should be the data row, NOT a header.
    assert first_line.startswith("r1\t")


def test_a_reloaded_model_carries_its_smiles_as_a_list(tmp_path):
    """cobra returns annotation values as lists for a model read from a
    file, which is what `save_ec_model` + `load_ec_model` produces. The
    SMILES has to survive that, and the ignore-list check has to not
    choke on it."""
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", 1.0, "beta")])],
        genes_per_rxn={"r1": ["g1"]},
        gene_sequences={"g1": "MASEQ"},
        met_smiles={"A": "C(C)O"},
    )
    for met in model.metabolites:
        if met.annotation.get("smiles"):
            met.annotation["smiles"] = [met.annotation["smiles"]]

    df = write_dlkcat_input(model, tmp_path / "DLKcat.tsv", _ignore_lists())
    assert len(df) == 1
    assert df.iloc[0]["smiles"] == "C(C)O"


def test_a_list_valued_smiles_is_still_matched_against_the_ignore_list(tmp_path):
    model = _ec_model(
        [("r1", [("A", -1.0, "alpha"), ("B", -1.0, "beta"), ("C", 1.0, "gamma")])],
        genes_per_rxn={"r1": ["g1"]},
        gene_sequences={"g1": "MASEQ"},
        met_smiles={"A": "SMI_A", "B": "O"},
    )
    for met in model.metabolites:
        if met.annotation.get("smiles"):
            met.annotation["smiles"] = [met.annotation["smiles"]]

    df = write_dlkcat_input(
        model, tmp_path / "DLKcat.tsv", _ignore_lists(ignore_smiles=["O"]),
    )
    assert set(df["substrate"]) == {"alpha"}
