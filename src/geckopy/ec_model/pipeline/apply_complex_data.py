"""Apply ComplexPortal subunit stoichiometries to ec.rxn_enz_mat.

Ported from GECKO MATLAB: src/change_model/applyComplexData.m.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...databases import (
    ComplexPortalEntry,
    load_complex_portal_json,
)
from .apply_kcat import apply_kcat_constraints

if TYPE_CHECKING:
    from ..ec_model import EcModel


logger = logging.getLogger(__name__)


def apply_complex_data(
    model: "EcModel",
    *,
    path: Optional[Path] = None,
    complex_data: Optional[list[ComplexPortalEntry]] = None,
    apply: bool = True,
    min_match_to_propose: float = 0.75,
) -> None:
    """Update ec.rxn_enz_mat with subunit stoichiometries from ComplexPortal.

    For each catalyzed reaction in ``ec.rxns``, look up its enzyme set
    in the ComplexPortal data. Three outcomes:

    Exact match
        The model's enzyme set equals a complex's protein list, with no
        extras on either side. Apply the complex's stoichiometry.

    Proposed
        Superset: every model enzyme is in a complex that lists additional
        subunits not in the model (the complex with the fewest extra
        subunits is reported). Partial: ``min_match_to_propose`` (default
        0.75) of the model enzymes are in a complex that has no subunits
        beyond the model's number, with extras or substitutions on the model
        side (the complex with the highest match is reported). A reaction
        can get one proposal of each kind, as in GECKO's
        ``applyComplexData``. Logged as WARNING; not applied.

    No match
        Skipped silently.

    Parameters
    ----------
    model
        An EcModel produced by ``make_ec_model``. Mutated in place.
    path
        Path to a ComplexPortal JSON file. If None and complex_data is
        also None, defaults to
        ``adapter.params.path / "data" / "ComplexPortal.json"``.
    complex_data
        Pre-loaded list of ComplexPortalEntry. Overrides ``path``.
    apply
        If True (default), call ``apply_kcat_constraints`` afterward.
        Needed because changing ``rxn_enz_mat`` invalidates the
        existing kcat coefficients.
    min_match_to_propose
        Minimum fraction of model enzymes in a complex for it to be
        reported as a proposal. MATLAB hardcodes 0.75.

    Raises
    ------
    FileNotFoundError
        If neither ``complex_data`` nor a valid ``path`` is provided.
    """
    if complex_data is None:
        if path is None:
            from ...adapter import resolve_adapter
            adapter = resolve_adapter(
                model,
                purpose="apply_complex_data needs a ComplexPortal JSON "
                "(pass `path=` explicitly, or rely on the adapter's "
                "default `<path>/data/ComplexPortal.json`)",
            )
            path = adapter.params.path / "data" / "ComplexPortal.json"
        complex_data = load_complex_portal_json(path)

    if not complex_data:
        logger.info(
            "apply_complex_data: ComplexPortal data is empty; nothing to do."
        )
        return

    enzyme_index = {e: i for i, e in enumerate(model.ec.enzymes)}

    # Build per-complex protein sets (full sets, including proteins not in
    # the model) and the corresponding stoichiometries. We index by complex
    # id to avoid building a wide sparse matrix.
    complex_proteins: list[set[str]] = []
    complex_stoich: list[dict[str, float]] = []
    for entry in complex_data:
        stoich = list(entry.stoichiometry)
        # MATLAB convention: all-zero stoichiometry -> all ones.
        if stoich and all(s == 0 for s in stoich):
            stoich = [1] * len(stoich)
        # Pad/truncate to align with protein_ids.
        if len(stoich) < len(entry.protein_ids):
            stoich = stoich + [1] * (len(entry.protein_ids) - len(stoich))
        prots = list(entry.protein_ids)
        complex_proteins.append(set(prots))
        complex_stoich.append({p: float(s) for p, s in zip(prots, stoich)})

    # For each protein in any complex, list which complexes mention it.
    # Used to find candidate complexes for a reaction quickly.
    protein_to_complexes: dict[str, list[int]] = {}
    for ci, prots in enumerate(complex_proteins):
        for p in prots:
            protein_to_complexes.setdefault(p, []).append(ci)

    rxn_enz_csr = model.ec.rxn_enz_mat.tocsr()
    rxn_enz_lil = rxn_enz_csr.tolil()

    n_found = 0
    n_proposed = 0

    for ri in range(model.ec.n_rxns):
        # Names of enzymes catalyzing this reaction in the model.
        enz_col_indices = rxn_enz_csr.getrow(ri).indices
        if enz_col_indices.size == 0:
            continue
        model_enzyme_names = {model.ec.enzymes[j] for j in enz_col_indices}

        # Candidate complexes: those that mention at least one model enzyme.
        candidate_indices: set[int] = set()
        for name in model_enzyme_names:
            for ci in protein_to_complexes.get(name, []):
                candidate_indices.add(ci)

        if not candidate_indices:
            continue

        mod_units = len(model_enzyme_names)

        exact_match: Optional[int] = None
        superset_candidate: Optional[tuple[int, float]] = None
        partial_candidate: Optional[tuple[int, float]] = None

        for ci in sorted(candidate_indices):
            complex_set = complex_proteins[ci]
            common = model_enzyme_names & complex_set
            match_units = len(common)
            total_units = len(complex_set)

            perc_match = match_units / mod_units
            total_match = total_units / mod_units

            if perc_match == 1.0 and total_match == 1.0:
                exact_match = ci
                break
            if perc_match == 1.0 and total_match > 1.0:
                if superset_candidate is None or total_match < superset_candidate[1]:
                    superset_candidate = (ci, total_match)
            elif (
                min_match_to_propose <= perc_match < 1.0
                and total_match <= 1.0
            ):
                if partial_candidate is None or perc_match > partial_candidate[1]:
                    partial_candidate = (ci, perc_match)

        if exact_match is not None:
            stoich = complex_stoich[exact_match]
            for prot_id, s in stoich.items():
                j = enzyme_index.get(prot_id)
                if j is not None:
                    rxn_enz_lil[ri, j] = s if s != 0 else 1.0
            n_found += 1
            continue

        if mod_units <= 1:
            continue

        proposals = [(c, kind) for c, kind in ((superset_candidate, "superset"), (partial_candidate, "partial"))
                     if c is not None]
        for (ci, score), match_kind in proposals:
            entry = complex_data[ci]
            n_proposed += 1
            logger.warning(
                "Proposed complex for reaction %s (%s match, %.0f%%): "
                "ComplexPortal '%s' has proteins %s; model has %s. "
                "Stoichiometry not applied; review manually.",
                model.ec.rxns[ri],
                match_kind,
                score * 100,
                entry.complex_id,
                entry.protein_ids,
                sorted(model_enzyme_names),
            )

    model.ec.rxn_enz_mat = rxn_enz_lil.tocsr()

    logger.info(
        "apply_complex_data: %d full match(es), %d proposed.",
        n_found, n_proposed,
    )

    if apply and n_found > 0:
        apply_kcat_constraints(model)
