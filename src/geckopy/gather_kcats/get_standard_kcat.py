"""Add a "standard" pseudoenzyme with median MW and kcat for reactions
without enzyme assignments.

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/getStandardKcat.m.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cobra
import numpy as np
from scipy import sparse

from ..ec_model.constants import POOL_ID
from ..ec_model.pipeline.protein_pool import _resolve_enzyme_compartment_id

if TYPE_CHECKING:
    from ..databases import UniprotDB
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


_STANDARD_NAME = "standard"
_STANDARD_MET_ID = "prot_standard"
_STANDARD_USAGE_RXN_ID = "usage_prot_standard"


def assign_standard_kcat(
    model: "EcModel",
    uniprot_db: "UniprotDB",
    *,
    threshold: int = 10,
    fill_zero_kcat: bool = True,
) -> None:
    """Add a "standard" pseudoenzyme to ``model`` and assign it to
    reactions without enzyme constraints.

    Ported from GECKO MATLAB:
    src/geckomat/gather_kcats/getStandardKcat.m.

    A standard MW (median of all proteins in ``uniprot_db``) and a
    standard kcat (median of all non-zero, non-NaN ``model.ec.kcat``
    values, or a per-subsystem median when the subsystem has at least
    ``threshold`` reactions with a real kcat) are computed. Reactions
    that lack a GPR rule, or have a GPR but no entry in
    ``model.ec.rxns``, are added to ``model.ec`` with this standard
    pseudoenzyme.

    The function is idempotent: previous "standard" entries
    (identified by ``model.ec.source == "standard"``) are stripped
    before re-applying.

    Filtering: exchange, transport (same metabolite name across
    compartments), pseudo (name contains "pseudoreaction"), SLIME
    (name contains "SLIME rxn"), spontaneous (via
    ``adapter.get_spontaneous_reactions(model)``), and reactions
    listed in ``adapter.params.path/data/pseudoRxns.tsv`` are
    excluded from receiving the standard pseudoenzyme.

    With ``fill_zero_kcat=True`` (default), reactions whose existing
    ``ec.kcat`` is 0 or NaN have their kcat replaced with
    ``standard_kcat`` and source set to ``"standard"``. They keep
    their original enzyme assignment.

    The function mutates ``model`` in place and returns ``None``; the
    standard MW/kcat and the counts of reactions touched are emitted
    via ``logger.info`` instead.

    Parameters
    ----------
    model
        EcModel with ``model.ec`` already allocated, the protein pool
        machinery already set up (``prot_pool`` metabolite present in
        the full-model branch), and ``model.adapter`` set.
        Mutated in place.
    uniprot_db
        Pre-loaded UniprotDB; used only for the median MW.
    threshold
        Minimum number of reactions with a real kcat in a subsystem
        before a subsystem-specific median kcat is used. Subsystems
        with fewer such reactions fall back to the global standard
        kcat.
    fill_zero_kcat
        Whether to replace existing 0/NaN ``ec.kcat`` entries with
        the standard kcat.

    Raises
    ------
    ValueError
        If ``model.adapter`` is None.
    """
    from ..adapter import resolve_adapter
    adapter = resolve_adapter(
        model,
        purpose="assign_standard_kcat reads params.enzyme_comp and the "
        "spontaneous-reactions list from the adapter",
    )

    standard_mw = _compute_standard_mw(uniprot_db)
    standard_kcat = _compute_standard_kcat(model.ec.kcat)
    subsystem_kcats = _compute_subsystem_kcats(model, threshold)

    rxns_missing = _find_reactions_missing_enzyme(model)

    custom_ignored = _load_custom_pseudo_rxns(adapter)
    ignore = _classify_reactions_to_ignore(model, custom_ignored)
    rxns_missing = [r for r in rxns_missing if r not in ignore]

    _remove_previous_standard(model)
    _add_standard_pseudoenzyme(model, standard_mw)
    rxns_added = _assign_standard_kcat_to_missing(
        model, rxns_missing, subsystem_kcats, standard_kcat,
    )

    if fill_zero_kcat:
        rxns_filled = _fill_zero_kcats(model, standard_kcat)
    else:
        rxns_filled = []

    logger.info(
        "assign_standard_kcat: standard MW = %.4g g/mmol, standard kcat = "
        "%.4g 1/s. Assigned standard pseudoenzyme to %d reaction(s); "
        "filled %d zero/NaN kcat entry(ies).",
        standard_mw, standard_kcat, len(rxns_added), len(rxns_filled),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _compute_standard_mw(uniprot_db: "UniprotDB") -> float:
    """Median MW (g/mmol) of UniProt proteins, NaN-safe."""
    mw = uniprot_db.mw
    if mw.size == 0:
        return float("nan")
    finite = mw[~np.isnan(mw)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _compute_standard_kcat(kcat: np.ndarray) -> float:
    """Median 1/s of non-zero, non-NaN ``ec.kcat`` entries."""
    if kcat.size == 0:
        return float("nan")
    nonzero = kcat[kcat > 0]
    finite = nonzero[~np.isnan(nonzero)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _first_subsystem(rxn: cobra.Reaction) -> str:
    """Return the first subsystem token (split on `;`) or empty string.

    Tolerates both string (MATLAB-style ``;``-joined) and list
    (cobra-py YAML round-trip) representations of ``rxn.subsystem``.
    """
    sub = rxn.subsystem
    if not sub:
        return ""
    if isinstance(sub, (list, tuple)):
        sub = sub[0] if sub else ""
    if not isinstance(sub, str):
        sub = str(sub)
    return sub.split(";")[0].strip()


def _ec_rxn_to_cobra_id(model: "EcModel", ec_rxn_id: str) -> str:
    return ec_rxn_id[4:] if model.ec.gecko_light else ec_rxn_id


def _compute_subsystem_kcats(
    model: "EcModel", threshold: int,
) -> dict[str, float]:
    """Per-subsystem median kcat for subsystems with >= threshold reactions.

    Uses the median (like the global standard kcat) rather than the
    arithmetic mean: kcats span orders of magnitude, so the mean is
    dominated by the largest member. Subsystems with fewer reactions are
    omitted; the caller falls back to the global standard kcat.
    """
    sub_kcats: dict[str, list[float]] = defaultdict(list)

    for ec_rxn_id, kcat in zip(model.ec.rxns, model.ec.kcat):
        # Only reactions with a real kcat contribute.
        if kcat == 0 or np.isnan(kcat):
            continue
        cobra_rxn_id = _ec_rxn_to_cobra_id(model, ec_rxn_id)
        try:
            rxn = model.reactions.get_by_id(cobra_rxn_id)
        except KeyError:
            continue
        sub = _first_subsystem(rxn)
        if not sub:
            continue
        sub_kcats[sub].append(float(kcat))

    return {
        sub: float(np.median(vals))
        for sub, vals in sub_kcats.items()
        if len(vals) >= threshold
    }


def _find_reactions_missing_enzyme(model: "EcModel") -> list[str]:
    """Reactions without GPR, OR with GPR but no entry in ec.rxns."""
    if model.ec.gecko_light:
        ec_rxn_set = {r[4:] for r in model.ec.rxns}
    else:
        ec_rxn_set = set(model.ec.rxns)

    no_gpr: list[str] = []
    has_gpr_no_ec: list[str] = []
    for rxn in model.reactions:
        if rxn.id.startswith("usage_prot_"):
            continue
        if not rxn.gene_reaction_rule:
            no_gpr.append(rxn.id)
        elif rxn.id not in ec_rxn_set:
            has_gpr_no_ec.append(rxn.id)
    return no_gpr + has_gpr_no_ec


def _detect_transport_reactions(model: "EcModel") -> set[str]:
    """A transport reaction has the same metabolite name appearing in
    different compartments."""
    transport_ids: set[str] = set()
    for rxn in model.reactions:
        comps_per_name: dict[str, set[str]] = defaultdict(set)
        for met in rxn.metabolites:
            comps_per_name[met.name].add(met.compartment)
        if any(len(comps) > 1 for comps in comps_per_name.values()):
            transport_ids.add(rxn.id)
    return transport_ids


def _classify_reactions_to_ignore(
    model: "EcModel", custom_rxns: set[str],
) -> set[str]:
    """Reactions that should not receive the standard pseudoenzyme."""
    ignore: set[str] = set(custom_rxns)

    for rxn in model.reactions:
        if rxn.boundary:
            ignore.add(rxn.id)
        name_lower = (rxn.name or "").lower()
        if "pseudoreaction" in name_lower or "slime rxn" in name_lower:
            ignore.add(rxn.id)

    ignore.update(_detect_transport_reactions(model))

    if model.adapter is not None:
        try:
            spontaneous = model.adapter.get_spontaneous_reactions(model)
            ignore.update(spontaneous or [])
        except Exception:
            pass

    return ignore


def _load_custom_pseudo_rxns(adapter) -> set[str]:
    """Read ``data/pseudoRxns.tsv`` from the adapter's project folder."""
    candidate = Path(adapter.params.path) / "data" / "pseudoRxns.tsv"
    if not candidate.is_file():
        return set()
    rxns: set[str] = set()
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rxn_id = line.split("\t")[0].strip()
        if rxn_id:
            rxns.add(rxn_id)
    return rxns


def _remove_previous_standard(model: "EcModel") -> None:
    """Strip out entries marked source='standard' from a prior run."""
    n = model.ec.n_rxns
    if n == 0:
        return

    std_col = (
        model.ec.enzymes.index(_STANDARD_NAME)
        if _STANDARD_NAME in model.ec.enzymes else None
    )

    rxn_enz_dense = (
        model.ec.rxn_enz_mat.toarray() if std_col is not None else None
    )

    rows_to_remove: list[int] = []
    rows_to_reset: list[int] = []
    for i in range(n):
        if model.ec.source[i] != "standard":
            continue
        if std_col is not None and rxn_enz_dense[i, std_col] > 0:
            rows_to_remove.append(i)
        else:
            rows_to_reset.append(i)

    for i in rows_to_reset:
        model.ec.kcat[i] = 0.0
        model.ec.source[i] = ""

    if rows_to_remove:
        keep = np.ones(n, dtype=bool)
        keep[rows_to_remove] = False
        model.ec.rxns = [r for r, k in zip(model.ec.rxns, keep) if k]
        model.ec.kcat = model.ec.kcat[keep]
        model.ec.source = [s for s, k in zip(model.ec.source, keep) if k]
        model.ec.notes = [n_ for n_, k in zip(model.ec.notes, keep) if k]
        model.ec.eccodes = [c for c, k in zip(model.ec.eccodes, keep) if k]
        model.ec.rxn_enz_mat = model.ec.rxn_enz_mat[keep, :].tocsr()


def _add_standard_pseudoenzyme(
    model: "EcModel", standard_mw: float,
) -> None:
    """Add the standard gene/met/usage rxn (if not already there) and
    extend the ec per-enzyme fields by one entry."""
    if _STANDARD_NAME in model.ec.enzymes:
        return

    if not model.ec.gecko_light:
        comp_id = _resolve_enzyme_compartment_id(
            model, model.adapter.params.enzyme_comp
        )
        prot_std = cobra.Metabolite(
            _STANDARD_MET_ID, name=_STANDARD_MET_ID, compartment=comp_id,
        )
        prot_std.notes["note"] = "Standard enzyme-usage pseudometabolite"
        model.add_metabolites([prot_std])

        pool_met = model.metabolites.get_by_id(POOL_ID)
        usage_rxn = cobra.Reaction(
            _STANDARD_USAGE_RXN_ID, name=_STANDARD_USAGE_RXN_ID,
        )
        usage_rxn.lower_bound = 0.0
        usage_rxn.upper_bound = 1000.0
        usage_rxn.add_metabolites({pool_met: -1.0, prot_std: 1.0})
        usage_rxn.gene_reaction_rule = _STANDARD_NAME
        model.add_reactions([usage_rxn])

    model.ec.genes = list(model.ec.genes) + [_STANDARD_NAME]
    model.ec.enzymes = list(model.ec.enzymes) + [_STANDARD_NAME]
    model.ec.mw = np.append(model.ec.mw, standard_mw)
    model.ec.sequence = list(model.ec.sequence) + [""]
    if model.ec.concs.size == len(model.ec.genes) - 1:
        model.ec.concs = np.append(model.ec.concs, np.nan)

    n_rxns, _ = model.ec.rxn_enz_mat.shape
    new_col = sparse.csr_matrix((n_rxns, 1), dtype=float)
    model.ec.rxn_enz_mat = sparse.hstack(
        [model.ec.rxn_enz_mat, new_col], format="csr",
    )


def _assign_standard_kcat_to_missing(
    model: "EcModel",
    rxn_ids: list[str],
    subsystem_kcats: dict[str, float],
    standard_kcat: float,
) -> list[str]:
    """Append ec entries for the listed reactions, each pointing to the
    standard pseudoenzyme."""
    if not rxn_ids:
        return []

    std_col = model.ec.enzymes.index(_STANDARD_NAME)

    kcats_to_add: list[float] = []
    for rxn_id in rxn_ids:
        rxn = model.reactions.get_by_id(rxn_id)
        sub = _first_subsystem(rxn)
        kcat = (
            subsystem_kcats.get(sub, standard_kcat)
            if sub
            else standard_kcat
        )
        if kcat == 0:  # subsystem had no real kcat to average -> standard
            kcat = standard_kcat
        kcats_to_add.append(kcat)

    if model.ec.gecko_light:
        new_ec_rxns = [f"001_{r}" for r in rxn_ids]
    else:
        new_ec_rxns = list(rxn_ids)

    n_new = len(rxn_ids)
    model.ec.rxns = list(model.ec.rxns) + new_ec_rxns
    model.ec.kcat = np.concatenate(
        [model.ec.kcat, np.array(kcats_to_add, dtype=float)]
    )
    model.ec.source = list(model.ec.source) + ["standard"] * n_new
    model.ec.notes = list(model.ec.notes) + [""] * n_new
    model.ec.eccodes = list(model.ec.eccodes) + [""] * n_new

    n_enz = model.ec.n_enzymes
    new_rows = sparse.lil_matrix((n_new, n_enz), dtype=float)
    for i in range(n_new):
        new_rows[i, std_col] = 1.0
    model.ec.rxn_enz_mat = sparse.vstack(
        [model.ec.rxn_enz_mat, new_rows.tocsr()], format="csr",
    )

    return list(rxn_ids)


def _fill_zero_kcats(model: "EcModel", standard_kcat: float) -> list[str]:
    """Replace unset ``ec.kcat`` entries (0 or NaN) with ``standard_kcat``
    and mark their source as 'standard'. Returns the list of rxn IDs
    filled."""
    if model.ec.kcat.size == 0:
        return []
    unset_mask = (model.ec.kcat == 0) | np.isnan(model.ec.kcat)
    if not unset_mask.any():
        return []
    model.ec.kcat[unset_mask] = standard_kcat
    indices = np.where(unset_mask)[0]
    for i in indices:
        model.ec.source[int(i)] = "standard"
    return [model.ec.rxns[int(i)] for i in indices]


def get_standard_kcat(
    model: "EcModel",
    uniprot_db: "UniprotDB",
    *,
    threshold: int = 10,
    fill_zero_kcat: bool = True,
) -> None:
    """Deprecated alias for :func:`assign_standard_kcat`.

    Kept for backward compatibility with the original MATLAB name.
    Will be removed in a future release; switch to
    ``assign_standard_kcat``.
    """
    import warnings

    warnings.warn(
        "get_standard_kcat is deprecated; use assign_standard_kcat "
        "instead. The old name will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return assign_standard_kcat(
        model, uniprot_db,
        threshold=threshold, fill_zero_kcat=fill_zero_kcat,
    )
