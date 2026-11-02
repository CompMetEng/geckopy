"""Apply user-curated custom kcats from a TSV file.

Ported from GECKO MATLAB: src/geckomat/change_model/applyCustomKcats.m.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .apply_kcat import apply_kcat_constraints

if TYPE_CHECKING:
    from ..ec_model import EcModel


logger = logging.getLogger(__name__)

_EXP_SUFFIX_REGEX = re.compile(r"_EXP_\d+$")
_SOURCE_TAG = "custom"
_NOTES_SEPARATOR = ", "


def apply_custom_kcats(
    model: "EcModel",
    path: Optional[Path] = None,
    *,
    apply: bool = True,
) -> None:
    """Apply user-curated kcats from a customKcats.tsv file.

    Reads a tab-separated file with seven columns and one header line:

        proteins | genes | gene_name | kcat | rxns | notes | stoicho

    Each row is interpreted in one of three modes, determined by which
    of the ``proteins`` and ``rxns`` columns are populated:

    Mode A (rxns only, proteins empty)
        Apply the kcat to every entry of ``ec.rxns`` whose ID, after
        stripping any ``_EXP_<n>`` suffix, equals one of the listed
        reaction IDs (comma-separated). A list in double quotes, as a
        spreadsheet exports it, is read without the quotes. Reverse
        direction is honored: ``"R2"`` matches ``R2_EXP_*`` but not
        ``R2_REV_*``.

    Mode B (proteins only, rxns empty)
        Candidate reactions are those catalyzed by any listed protein.
        For each candidate, the kcat is applied only if the candidate's
        enzyme set exactly matches the listed proteins (full match).
        Partial matches (>= 50% but < 100%) are logged for manual review.

    Mode C (both populated)
        Same matching rule as Mode B, but candidates are restricted to
        reactions matching the listed reaction IDs (with the same
        ``_EXP_`` stripping as Mode A).

    Notes from the ``notes`` column are appended to ``ec.notes`` for
    each updated reaction (separated by ``, `` from any existing note).
    The ``ec.source`` is set to ``"custom"``. The ``stoicho`` column is
    parsed but not applied; a warning is logged if any row has
    non-trivial stoichiometry. The ``genes`` and ``gene_name`` columns
    are read but not used.

    Rows that fail to resolve to any reaction, or whose enzyme match
    is only partial, are logged at WARNING level for manual review
    rather than raised as errors.

    Parameters
    ----------
    model
        An EcModel produced by ``make_ec_model``. Mutated in place.
    path
        Path to the TSV. If None, defaults to
        ``adapter.params.path / "data" / "customKcats.tsv"``.
    apply
        If True (default), call ``apply_kcat_constraints`` after
        updating ``ec.kcat`` so the new values reflect in the S matrix.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has fewer than 7 tab-delimited columns in the header.
    """
    if path is None:
        from ...adapter import resolve_adapter
        adapter = resolve_adapter(
            model,
            purpose="apply_custom_kcats needs a customKcats.tsv path "
            "(pass `path=` explicitly, or rely on the adapter's "
            "default `<path>/data/customKcats.tsv`)",
        )
        path = adapter.params.path / "data" / "customKcats.tsv"
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"customKcats.tsv not found: {path}")

    rows = _parse_custom_kcats_tsv(path)

    ec_rxn_ids = model.ec.rxns
    no_suffix = [_EXP_SUFFIX_REGEX.sub("", r) for r in ec_rxn_ids]
    enzyme_index = {enz: i for i, enz in enumerate(model.ec.enzymes)}

    updated_indices: set[int] = set()

    for row in rows:
        proteins, rxns_in, kcat, notes, stoicho = (
            row["proteins"], row["rxns"], row["kcat"], row["notes"],
            row["stoicho"],
        )

        # A blank kcat cell parses to NaN. Writing it would mark the
        # reaction "unassigned" and silently clear an existing kcat in
        # apply_kcat_constraints, so skip the row instead.
        if math.isnan(kcat):
            logger.warning(
                "Row %d: kcat is blank; row skipped (existing kcat kept).",
                row["index"],
            )
            continue
        if kcat < 0:
            logger.warning(
                "Row %d: kcat %g is negative; row skipped.",
                row["index"], kcat,
            )
            continue

        if stoicho and any(s.strip() not in ("", "1") for s in stoicho.split("+")):
            logger.warning(
                "Row %d: stoichiometry %r is ignored. Subunit counts must "
                "be set via a separate function (not yet available).",
                row["index"], stoicho,
            )

        # Mode A: rxns only.
        if not proteins and rxns_in:
            matched = _resolve_rxn_ids(rxns_in, no_suffix, ec_rxn_ids)
            if not matched:
                logger.warning(
                    "Row %d (rxns=%r): no reactions matched.",
                    row["index"], rxns_in,
                )
                continue
            _apply_to_indices(model, matched, kcat, notes, updated_indices)
            continue

        # Modes B and C: proteins populated.
        if not proteins:
            logger.warning(
                "Row %d: both 'proteins' and 'rxns' columns are empty; "
                "skipping.", row["index"],
            )
            continue

        prot_list = [p.strip() for p in proteins.split("+") if p.strip()]
        prot_indices: list[int] = []
        for prot in prot_list:
            idx = enzyme_index.get(prot)
            if idx is None:
                logger.warning(
                    "Row %d: protein %r not found in ec.enzymes.",
                    row["index"], prot,
                )
                prot_indices = []
                break
            prot_indices.append(idx)
        if not prot_indices:
            continue

        # Determine candidate ec.rxns indices.
        if rxns_in:
            candidate_indices = _resolve_rxn_ids(
                rxns_in, no_suffix, ec_rxn_ids
            )
        else:
            mat_csc = model.ec.rxn_enz_mat.tocsc()
            seen: set[int] = set()
            for pidx in prot_indices:
                col = mat_csc.getcol(pidx)
                seen.update(col.nonzero()[0].tolist())
            candidate_indices = sorted(seen)

        if not candidate_indices:
            logger.warning(
                "Row %d (proteins=%r, rxns=%r): no candidate reactions.",
                row["index"], proteins, rxns_in,
            )
            continue

        # Apply full-match gate.
        prot_set = set(prot_indices)
        mat_csr = model.ec.rxn_enz_mat.tocsr()
        full_match: list[int] = []
        partial_match: list[tuple[int, float]] = []
        for cand in candidate_indices:
            rxn_enzymes = set(mat_csr.getrow(cand).indices.tolist())
            common = prot_set & rxn_enzymes
            if len(common) == len(prot_set) == len(rxn_enzymes):
                full_match.append(cand)
            elif rxn_enzymes:
                denom = max(len(prot_set), len(rxn_enzymes))
                fraction = len(common) / denom
                if fraction >= 0.5:
                    partial_match.append((cand, fraction))

        if full_match:
            _apply_to_indices(model, full_match, kcat, notes, updated_indices)

        for cand_idx, fraction in partial_match:
            logger.warning(
                "Row %d (proteins=%r): partial match (%.0f%%) with reaction "
                "%s; not applied. Curate manually.",
                row["index"], proteins, fraction * 100,
                ec_rxn_ids[cand_idx],
            )

        if not full_match and not partial_match:
            logger.warning(
                "Row %d (proteins=%r, rxns=%r): no full or partial match.",
                row["index"], proteins, rxns_in,
            )

    if not updated_indices:
        logger.warning(
            "apply_custom_kcats: no reactions were updated. Check that the "
            "IDs and proteins in %s match the model.", path,
        )
        return

    if apply:
        updated_rxn_ids = sorted(ec_rxn_ids[i] for i in updated_indices)
        apply_kcat_constraints(model, update_rxns=updated_rxn_ids)


def _parse_custom_kcats_tsv(path: Path) -> list[dict]:
    """Parse the seven-column TSV into a list of row dicts (keys:
    index, proteins, kcat, rxns, notes, stoicho).
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if len(header) < 7:
            raise ValueError(
                f"{path} header has {len(header)} columns; expected at least 7."
            )
        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            while len(parts) < 7:
                parts.append("")
            try:
                kcat = float(parts[3]) if parts[3].strip() else float("nan")
            except ValueError:
                logger.warning(
                    "Row %d in %s: kcat %r is not numeric; row skipped.",
                    line_no - 1, path, parts[3],
                )
                continue

            rows.append({
                "index": line_no - 1,  # 1-based row index excluding header
                "proteins": parts[0].strip(),
                "kcat": kcat,
                "rxns": _unquote(parts[4].strip()),
                "notes": parts[5].strip(),
                "stoicho": parts[6].strip(),
            })
    return rows


def _unquote(field: str) -> str:
    """Strip one pair of double quotes around a field, as MATLAB's ``%q`` does.

    A spreadsheet exports a reaction list such as ``R3,R5`` as ``"R3,R5"``;
    a doubled quote inside the field stands for one quote.
    """
    if len(field) >= 2 and field.startswith('"') and field.endswith('"'):
        return field[1:-1].replace('""', '"')
    return field


def _resolve_rxn_ids(
    rxns_field: str, no_suffix: list[str], ec_rxn_ids: list[str],
) -> list[int]:
    """Mode A / Mode C: resolve a comma-separated rxns field to ec-row indices."""
    requested = [r.strip() for r in rxns_field.split(",") if r.strip()]
    matched: list[int] = []
    seen: set[int] = set()
    for req in requested:
        for i, base in enumerate(no_suffix):
            if base == req and i not in seen:
                matched.append(i)
                seen.add(i)
    return matched


def _apply_to_indices(
    model: "EcModel",
    indices: list[int],
    kcat: float,
    notes: str,
    updated_indices: set[int],
) -> None:
    """Write kcat, source, and notes to ec.* for the given indices."""
    for idx in indices:
        model.ec.kcat[idx] = kcat
        model.ec.source[idx] = _SOURCE_TAG
        if notes:
            existing = model.ec.notes[idx]
            if existing:
                model.ec.notes[idx] = existing + _NOTES_SEPARATOR + notes
            else:
                model.ec.notes[idx] = notes
        updated_indices.add(idx)
