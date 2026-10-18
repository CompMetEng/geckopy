"""Write a DLKcat-compatible TSV input file for an ec model.

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/writeDLKcatInput.m.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from ..databases.pubchem import met_smiles as read_met_smiles

if TYPE_CHECKING:
    from ..databases.dlkcat_ignore_lists import DLKcatIgnoreLists
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


_OUTPUT_COLUMNS = ["rxn_id", "gene", "substrate", "smiles", "sequence", "kcat"]
_PAIR_COLUMNS = ["rxn_id", "gene", "substrate", "smiles", "sequence", "met_id"]
_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z]+")


def write_dlkcat_input(
    model: "EcModel",
    output_path: str | Path,
    ignore_lists: "DLKcatIgnoreLists",
    *,
    ec_rxns: Optional[Iterable[str]] = None,
    only_with_smiles: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write a DLKcat-compatible TSV input file.

    Ported from GECKO MATLAB:
    src/geckomat/gather_kcats/writeDLKcatInput.m.

    For each selected ec.rxn, enumerate (substrate, gene-subunit)
    pairs from the model's stoichiometry and rxn_enz_mat. Each row
    of output corresponds to one ``(rxn_id, gene, substrate)``
    triple, plus the substrate's SMILES and the gene's protein
    sequence. The output file is tab-separated with no header,
    matching DLKcat's expected input format.

    Filtering steps:

    * Metabolites whose normalized name is in
      ``ignore_lists.ignore_names`` (lowercased, alphanumeric-only)
      are dropped.
    * Metabolites whose ``annotation['smiles']`` matches one of
      ``ignore_lists.ignore_smiles`` are dropped.
    * Metabolites whose ID starts with ``prot_`` (protein-usage
      pseudometabolites added by earlier pipeline stages) are
      dropped.
    * Currency pairs: when both halves of a pair appear in the same
      reaction, both are removed UNLESS removing them would leave
      the reaction with no substrates.
    * If ``only_with_smiles`` is True, rows whose substrate has no
      SMILES are dropped from the output. Otherwise the SMILES is
      written as the literal string ``"None"``.

    SMILES are read from each metabolite's ``annotation['smiles']``,
    which cobra returns as a list for a model loaded from a file.

    Parameters
    ----------
    model
        EcModel with populated ``ec.rxns``, ``ec.genes``,
        ``ec.sequence``, and ``ec.rxn_enz_mat``.
    output_path
        Path to the output TSV. Refuses to overwrite an existing
        file unless ``overwrite=True``.
    ignore_lists
        Pre-loaded ``DLKcatIgnoreLists`` (typically from
        ``load_dlkcat_ignore_lists``).
    ec_rxns
        Optional iterable of reaction IDs (must be in
        ``model.ec.rxns``). ``None`` means all.
    only_with_smiles
        When True (default), drop rows whose substrate has no
        SMILES. When False, write ``"None"`` in the SMILES column
        instead.
    overwrite
        Allow overwriting an existing output file.

    Returns
    -------
    pandas.DataFrame
        The rows that were written, with columns
        ``rxn_id, gene, substrate, smiles, sequence, kcat``. Useful
        for tests and inspection. The on-disk file has no header.

    Raises
    ------
    FileExistsError
        If ``output_path`` exists and ``overwrite=False``.
    ValueError
        If any ID in ``ec_rxns`` is not present in
        ``model.ec.rxns``, or if any selected reaction is missing
        from ``model.reactions``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set overwrite=True to replace."
        )

    pairs = extract_enzyme_substrate_pairs(
        model, ignore_lists, ec_rxns=ec_rxns, only_with_smiles=only_with_smiles,
    )
    df = pairs.copy()
    df["kcat"] = "NA"
    df = df[_OUTPUT_COLUMNS]

    _write_tsv(output_path, df)
    logger.info(
        "write_dlkcat_input: wrote %d row(s) to %s.", len(df), output_path,
    )
    return df


def extract_enzyme_substrate_pairs(
    model: "EcModel",
    ignore_lists: "DLKcatIgnoreLists",
    *,
    ec_rxns: Optional[Iterable[str]] = None,
    only_with_smiles: bool = True,
) -> pd.DataFrame:
    """Enumerate (reaction, gene, substrate, SMILES, sequence) rows for an ec model.

    Shared core of ``write_dlkcat_input`` and the OpenKineticsPredictor
    input builder. Performs the same filtering (ignored metabolites,
    ``prot_`` pseudometabolites, currency-metabolite pairs) and the same
    ``only_with_smiles`` handling, but does no file I/O.

    Parameters
    ----------
    model
        EcModel with ``ec.rxns``, ``ec.genes``, ``ec.sequence`` and
        ``ec.rxn_enz_mat`` populated, and per-metabolite SMILES in
        ``annotation['smiles']``.
    ignore_lists
        Loaded ``DLKcatIgnoreLists``.
    ec_rxns
        Optional iterable of reaction IDs (must be in ``model.ec.rxns``).
        ``None`` means all.
    only_with_smiles
        Drop rows with no SMILES when True; otherwise write ``"None"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``rxn_id, gene, substrate, smiles, sequence, met_id``.
    """
    n_ec = model.ec.n_rxns
    if ec_rxns is None:
        ec_rxn_indices = list(range(n_ec))
    else:
        ec_rxns_list = list(ec_rxns)
        index_by_id = {rid: i for i, rid in enumerate(model.ec.rxns)}
        unknown = [rid for rid in ec_rxns_list if rid not in index_by_id]
        if unknown:
            preview = unknown[:5]
            raise ValueError(
                f"{len(unknown)} reaction ID(s) in ec_rxns are not present "
                f"in model.ec.rxns (examples: {preview})"
            )
        ec_rxn_indices = [index_by_id[rid] for rid in ec_rxns_list]

    if not ec_rxn_indices:
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    # Map ec.rxns to cobra reaction IDs (with gecko_light prefix).
    if model.ec.gecko_light:
        orig_rxn_ids = [r[4:] for r in model.ec.rxns]
    else:
        orig_rxn_ids = list(model.ec.rxns)
    selected_orig = [orig_rxn_ids[i] for i in ec_rxn_indices]

    cobra_rxn_id_set = {r.id for r in model.reactions}
    missing = [r for r in selected_orig if r not in cobra_rxn_id_set]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} reaction(s) in model.ec.rxns are not present "
            f"in model.reactions (examples: {preview})"
        )

    # Per-metabolite info in cobra order.
    metabolites = list(model.metabolites)
    met_ids = [m.id for m in metabolites]
    met_names = [m.name for m in metabolites]
    met_smiles = [read_met_smiles(m) for m in metabolites]
    met_normalized = [_normalize(n) for n in met_names]

    # Build the ignore mask.
    ignore_names_set = set(ignore_lists.ignore_names)
    ignore_smiles_set = {s for s in ignore_lists.ignore_smiles if s}
    ignore_mask = np.array([
        normalized in ignore_names_set
        or (smiles and smiles in ignore_smiles_set)
        or met_id.startswith("prot_")
        for normalized, smiles, met_id
        in zip(met_normalized, met_smiles, met_ids)
    ])

    # Build the stoichiometric matrix in cobra order.
    s_matrix = _build_s_matrix(model, metabolites)

    # Zero out ignored metabolites.
    s_lil = s_matrix.tolil()
    for i in np.where(ignore_mask)[0]:
        s_lil[i, :] = 0
    reduced_s = s_lil.tocsr()

    # Strip currency pairs.
    reduced_s = _strip_currency_pairs(
        reduced_s, met_normalized, ignore_lists.currency_pairs,
    )

    # Subset to selected reactions.
    rxn_id_to_col = {r.id: j for j, r in enumerate(model.reactions)}
    selected_cols = np.array([rxn_id_to_col[r] for r in selected_orig])
    sub_matrix = reduced_s[:, selected_cols].toarray()

    # (substrate_idx, local_rxn_idx) pairs where coefficient is negative.
    #
    # MATLAB's `[substrates, reactions] = find(clearedRedS < 0)` walks a
    # sparse matrix column-major, so the pairs come out reaction-major
    # (all substrates of reaction 1, then of reaction 2, ...). numpy's
    # `where` is row-major, which would group by substrate instead; the
    # transpose restores MATLAB's row order in the written file.
    local_rxn_idx, substrate_idx = np.where(sub_matrix.T < 0)

    if len(substrate_idx) == 0:
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    rxn_enz_mat = model.ec.rxn_enz_mat.tocsr()

    rows: list[dict[str, str]] = []
    for sub_i, local_i in zip(substrate_idx, local_rxn_idx):
        ec_idx = ec_rxn_indices[local_i]
        gene_indices = rxn_enz_mat[ec_idx].indices
        for gene_idx in gene_indices:
            rows.append({
                "rxn_id": model.ec.rxns[ec_idx],
                "gene": model.ec.genes[gene_idx]
                        if gene_idx < len(model.ec.genes) else "",
                "substrate": met_names[sub_i],
                "smiles": met_smiles[sub_i],
                "sequence": model.ec.sequence[gene_idx]
                            if gene_idx < len(model.ec.sequence) else "",
                "met_id": met_ids[sub_i],
            })

    df = pd.DataFrame(rows, columns=_PAIR_COLUMNS)

    if only_with_smiles:
        df = df[df["smiles"].astype(str) != ""].reset_index(drop=True)
    else:
        empty_mask = df["smiles"].astype(str) == ""
        df.loc[empty_mask, "smiles"] = "None"

    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize(name: str) -> str:
    """Lowercase ``name`` with all non-alphanumeric characters removed."""
    return _NORMALIZE_RE.sub("", name).lower()


def _build_s_matrix(model, metabolites) -> sparse.csr_matrix:
    """Stoichiometric matrix in (metabolite, reaction) cobra order."""
    met_idx = {m.id: i for i, m in enumerate(metabolites)}
    reactions = list(model.reactions)
    s = sparse.lil_matrix(
        (len(metabolites), len(reactions)), dtype=float,
    )
    for j, rxn in enumerate(reactions):
        for met, coeff in rxn.metabolites.items():
            s[met_idx[met.id], j] = coeff
    return s.tocsr()


def _strip_currency_pairs(
    s_matrix: sparse.csr_matrix,
    met_normalized: list[str],
    currency_pairs: list[tuple[str, str]],
) -> sparse.csr_matrix:
    """For each currency pair, zero both halves in reactions that
    contain BOTH halves, unless removal would leave the reaction
    with no substrates."""
    if not currency_pairs:
        return s_matrix

    s = s_matrix.tolil()
    s_dense = s_matrix.toarray()  # for faster row/col reads in the inner loop

    for left_norm, right_norm in currency_pairs:
        left_idx = [
            i for i, n in enumerate(met_normalized) if n == left_norm
        ]
        right_idx = [
            i for i, n in enumerate(met_normalized) if n == right_norm
        ]
        if not left_idx or not right_idx:
            continue

        # Reactions containing any "left" met (any nonzero coeff).
        left_mask = (s_dense[left_idx, :] != 0).any(axis=0)
        right_mask = (s_dense[right_idx, :] != 0).any(axis=0)
        pair_rxns = np.where(left_mask & right_mask)[0]
        if len(pair_rxns) == 0:
            continue

        all_pair_indices = left_idx + right_idx

        for j in pair_rxns:
            col = s_dense[:, j].copy()
            col[all_pair_indices] = 0
            if (col < 0).any():
                # Remove the pair from this reaction.
                for i in all_pair_indices:
                    s[i, j] = 0
                    s_dense[i, j] = 0

    return s.tocsr()


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    """Write the DataFrame as tab-separated, no header."""
    df.to_csv(path, sep="\t", index=False, header=False)
