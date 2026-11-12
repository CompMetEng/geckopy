"""Bring GECKO 3 projects and ecModels to the GECKO 4 layout.

- :func:`convert_custom_kcats`: ``data/customKcats.tsv`` becomes the curated
  kcat source file ``data/kcats/custom.tsv``.
- :func:`upgrade_ec_model`: protein usage and pool reactions of a GECKO 3
  ecModel turned forward.
- :func:`upgrade_project` (``geckopy upgrade``): these and the other steps
  for a project folder.

GECKO (MATLAB) has the same as ``convertCustomKcats``, ``upgradeEcModel``
and ``upgradeGECKOproject``.
"""
from __future__ import annotations

import logging
import math
import re
import shutil
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .ec_model.constants import POOL_EXCHANGE_ID, POOL_ID
from .ec_model.pipeline.apply_kcat import apply_kcat_constraints
from .gather_kcats.kcat_source_format import format_kcat_source, parse_kcat_source

if TYPE_CHECKING:
    from .adapter import ModelAdapter
    from .ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_EXP_SUFFIX_REGEX = re.compile(r"_EXP_\d+$")
_NOTES_SEPARATOR = ", "

#: Renamed or replaced functions, as (old name, what to use).
RENAMED = (
    ("get_standard_kcat", "assign_standard_kcat"),
    ("get_kcat_across_isozymes", "fill_kcats_from_isozymes"),
    ("get_ec_from_database", "fill_eccodes_from_database"),
    ("get_ec_from_gem", "fill_eccodes_from_gem"),
    ("sigma_fitter", "fit_sigma"),
    ("constrain_flux_data", "apply_flux_data_constraints"),
    ("select_kcat_value", "apply_kcat_list"),
    ("merge_dlkcat_and_fuzzy_kcats", "merge_kcats"),
    ("fuzzy_kcat_matching", "kcats_from_brenda (or kcats_from_ec_tree), then assign_kcats"),
    ("read_dlkcat_output", 'kcats_from_dlkcat(via="file"), then assign_kcats'),
    ("run_dlkcat", 'kcats_from_dlkcat(via="docker"), then assign_kcats'),
    ("write_dlkcat_input", 'kcats_from_dlkcat(via="docker"), then assign_kcats'),
    ("parse_okp_output", "kcats_from_okp, then assign_kcats"),
    ("submit_open_kinetics_predictor", "kcats_from_okp, then assign_kcats"),
    ("fetch_open_kinetics_predictor", "kcats_from_okp, then assign_kcats"),
    ("build_okp_input_csv", "kcats_from_okp, then assign_kcats"),
    ("apply_custom_kcats", "convert_custom_kcats (data/kcats/custom.tsv), then assign_kcats"),
    ("truncate_values", "removed: nothing replaces it"),
    ("find_max_value", "removed: nothing replaces it"),
)


def convert_custom_kcats(
    model: "EcModel",
    *,
    custom_kcats: Optional[str | Path] = None,
    out_file: Optional[str | Path] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> tuple[Path, dict]:
    """Convert ``data/customKcats.tsv`` into ``data/kcats/custom.tsv``.

    The kcats are those GECKO 3 applied to the ecModel from this file
    (``applyCustomKcats``), including which of two entries for the same
    reaction is kept: one row per reaction, keyed by conventional reaction and
    enzyme set (see :func:`~geckopy.gather_kcats.ec_rxn_keys`), with the entry's notes in
    the note column. The ``stoicho`` column is left out: complex
    stoichiometry comes from ``apply_complex_data``. The conversion needs the
    full ecModel; ``custom.tsv`` applies to full and light ecModels.

    Parameters
    ----------
    model
        The full ecModel the custom kcats are for.
    custom_kcats, out_file
        The file to convert and the file to write (default:
        ``data/customKcats.tsv`` and ``data/kcats/custom.tsv`` in the
        project). An existing ``out_file`` is not overwritten.
    adapter
        Model adapter (default: ``model.adapter``).

    Returns
    -------
    (path, report)
        The file written, and a report with ``rows`` (rows written).
    """
    from .adapter import resolve_adapter
    from .gather_kcats.kcat_sources import _geckopy_version, ec_rxn_keys, format_number

    if model.ec.gecko_light:
        raise ValueError("convert_custom_kcats needs the full ecModel. "
                         "The custom.tsv written applies to light ecModels too.")
    adapter = resolve_adapter(model, adapter, purpose="convert_custom_kcats needs the project folder")
    project = Path(adapter.params.path)
    src = Path(custom_kcats) if custom_kcats is not None else project / "data" / "customKcats.tsv"
    out = Path(out_file) if out_file is not None else project / "data" / "kcats" / "custom.tsv"
    if not src.is_file():
        raise FileNotFoundError(f"No custom kcats file {src}.")
    if out.exists():
        raise FileExistsError(f"{out} exists and is not overwritten.")

    applied = model.copy()
    n = len(applied.ec.rxns)
    applied.ec.notes = [""] * n
    applied.ec.source = [""] * n
    _apply_custom_kcats(applied, path=src, apply=False)
    base, enzymes = ec_rxn_keys(applied)
    rows: dict[tuple[str, str], list[str]] = {}
    for i in range(n):
        if parse_kcat_source(applied.ec.source[i]).origin != "custom":
            continue
        note = re.sub(r"[\t\r\n]", " ", applied.ec.notes[i] or "")
        rows.setdefault((base[i], enzymes[i]), [base[i], enzymes[i], "", format_number(float(applied.ec.kcat[i])),
                                                "", "custom", "", "", note])
    ordered = sorted(rows.values(), key=lambda r: (r[0], r[1], r[2], r[5], r[3], r[4], r[6], r[7], r[8]))
    out.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = ["# source: custom", f"# generator: convert_custom_kcats (geckopy {_geckopy_version()})",
              "# parameters: {}", f"# data: {src.name}", "# kind: curated", f"# created: {created}",
              "rxn\tenzymes\tsubstrate\tkcat\tlog10_sd\tsource\tdetail\tec_code\tnote"]
    out.write_text("\n".join(header + ["\t".join(r) for r in ordered]) + "\n", encoding="utf-8", newline="\n")
    logger.info("%d custom kcats written to %s.", len(ordered), out)
    return out, {"rows": len(ordered)}


def upgrade_ec_model(model: "EcModel") -> list[str]:
    """Turn the protein usage and pool reactions of a GECKO 3 ecModel forward.

    GECKO 3 defined ``usage_prot_<id>`` (``prot_<id> -> prot_pool``) and
    ``prot_pool_exchange`` (``prot_pool ->``) in the reverse direction, used
    with negative flux; GECKO 4 defines them forward. Each reaction that is
    still reverse gets the opposite stoichiometry, bounds and objective
    coefficient; calling it again changes nothing. In place.
    :func:`~geckopy.load_ec_model` already does this for the files it
    reads; this is for models from other sources, such as SBML.

    Returns
    -------
    list of str
        The reactions that were turned.
    """
    if POOL_ID not in model.metabolites:
        return []
    pool = model.metabolites.get_by_id(POOL_ID)
    changed = []
    for rxn in model.reactions:
        coef = rxn.metabolites.get(pool, 0)
        old = (rxn.id.startswith("usage_prot_") and coef > 0) or (rxn.id == POOL_EXCHANGE_ID and coef < 0)
        if not old:
            continue
        objective = rxn.objective_coefficient
        rxn.add_metabolites({m: -2 * c for m, c in rxn.metabolites.items()})
        lb, ub = rxn.bounds
        rxn.bounds = (-ub, -lb)
        if objective:
            rxn.objective_coefficient = -objective
        changed.append(rxn.id)
    return changed


def upgrade_project(
    project: str | Path,
    *,
    dry_run: bool = True,
    ec_model: Optional["EcModel"] = None,
) -> pd.DataFrame:
    """Bring a GECKO 3 project to the GECKO 4 layout (``geckopy upgrade``).

    Lists, and with ``dry_run=False`` carries out:

    1. custom kcats: ``data/customKcats.tsv`` becomes ``data/kcats/custom.tsv``
       (:func:`convert_custom_kcats`);
    2. stored predictions: ``data/DLKcat.tsv`` and stored
       OpenKineticsPredictor results become kcat source files
       (``kcats_from_dlkcat``, ``kcats_from_okp`` with ``use_stored``),
       without running anything;
    3. saved ecModels: ``models/*.yml`` ecModels made with GECKO 3 get their
       protein reactions turned forward (:func:`upgrade_ec_model`);
    4. ``project_format = 4`` in ``model_adapter.toml``;
    5. scripts: lines of the project's ``.py`` files that call renamed or
       replaced functions are listed, not changed.

    Steps 1 and 2 need the full ecModel the custom kcats and predictions are
    for (``ec_model``); without it they are listed and skipped. A project
    with an adapter class instead of ``model_adapter.toml`` is converted by
    GECKO's ``upgradeGECKOproject``. A file that is replaced or converted is
    first moved to ``legacy/<yyyy-mm-dd>/``, keeping its place in the folder
    tree. Steps already done are left out, so it can run again.

    Returns
    -------
    pandas.DataFrame
        One row per step: ``step``, ``status`` (to do, done, skipped,
        failed, by hand) and ``note``.
    """
    from .adapter import ModelAdapter

    project = Path(project).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"No folder {project}.")
    apply = not dry_run
    status = "done" if apply else "to do"
    legacy = project / "legacy" / date.today().isoformat()
    toml = project / "model_adapter.toml"
    rows: list[tuple[str, str, str]] = []

    def keep_original(path: Path) -> None:
        target = legacy / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)

    adapter = None
    if not toml.is_file():
        rows.append(("adapter", "by hand", "no model_adapter.toml; convert an adapter class with GECKO's "
                     "upgradeGECKOproject, or start one with geckopy init"))
    else:
        adapter = ModelAdapter.from_folder(project)

    custom, custom_out = project / "data" / "customKcats.tsv", project / "data" / "kcats" / "custom.tsv"
    if custom.is_file() and not custom_out.exists():
        note = "data/customKcats.tsv -> data/kcats/custom.tsv"
        if ec_model is None or adapter is None:
            rows.append(("custom kcats", "skipped", note + "; give the ecModel they are for (ec_model)"))
        else:
            if apply:
                _, report = convert_custom_kcats(ec_model, custom_kcats=custom, out_file=custom_out, adapter=adapter)
                keep_original(custom)
                note += f" ({report['rows']} rows)"
            rows.append(("custom kcats", status, note))

    stored = _stored_predictions(project)
    if stored:
        note = "stored predictions -> data/kcats: " + ", ".join(stored)
        if ec_model is None or adapter is None:
            rows.append(("predictions", "skipped", note + "; give the ecModel they were made for (ec_model)"))
        elif not apply:
            rows.append(("predictions", status, note))
        else:
            from .gather_kcats import kcats_from_dlkcat, kcats_from_okp

            for name in stored:
                try:
                    if name == "dlkcat":
                        kcats_from_dlkcat(ec_model, adapter=adapter)
                    else:
                        kcats_from_okp(ec_model, method=name, use_stored=True, adapter=adapter)
                    rows.append(("predictions", status, f"stored predictions -> data/kcats: {name}"))
                except Exception as err:  # one failing source does not stop the others
                    rows.append(("predictions", "failed", f"{name}: {err}"))

    for path in sorted((project / "models").glob("*.yml")):
        if "prot_pool" not in path.read_text(encoding="utf-8", errors="ignore"):
            continue
        from .utilities.load_ec_model import load_ec_model
        from .utilities.save_ec_model import save_ec_model

        # load_ec_model turns GECKO 3 protein reactions forward, with a warning
        # that names how many; the file itself keeps the old direction.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = load_ec_model(path)
            except Exception as err:  # an ecModel this geckopy cannot read stops only its own row
                rows.append(("saved ecModel", "failed", f"{path.relative_to(project)}: {err}"))
                continue
        flipped = [re.search(r"(\d+) protein usage", str(w.message)) for w in caught]
        flipped = [int(m.group(1)) for m in flipped if m]
        changed = upgrade_ec_model(model)
        n_turned = sum(flipped) + len(changed)
        if not n_turned:
            continue
        if apply:
            keep_original(path)
            save_ec_model(model, path)
        rows.append(("saved ecModel", status, f"{path.relative_to(project)}: {n_turned} protein reactions "
                     "turned forward"))

    if toml.is_file() and not re.search(r"(?m)^\s*project_format\s*=", toml.read_text(encoding="utf-8")):
        if apply:
            toml.write_text("project_format = 4\n" + toml.read_text(encoding="utf-8"), encoding="utf-8")
        rows.append(("project format", status, "project_format = 4 in model_adapter.toml"))

    for path in sorted(project.rglob("*.py")):
        if legacy.parent in path.parents:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for old, new in RENAMED:
                if re.search(rf"(?<![\w.]){old}\s*\(", line) or re.search(rf"\bimport\b.*\b{old}\b", line):
                    rows.append(("script", "by hand", f"{path.relative_to(project)}:{lineno}: {old} -> {new}"))

    report = pd.DataFrame(rows, columns=["step", "status", "note"])
    if report.empty:
        logger.info("%s has the current GECKO 4 layout.", project)
    return report


def _stored_predictions(project: Path) -> list[str]:
    """Stored predictions without a source file: ``dlkcat`` and OKP methods."""
    from .gather_kcats.kcat_assignment import okp_file

    data = project / "data"
    found = []
    if (data / "DLKcat.tsv").is_file() and not (data / "kcats" / "dlkcat.tsv").exists():
        found.append("dlkcat")
    for out in sorted(data.glob("OKP_output*.csv")):
        job = data / ("OKP_job.txt" if out.name == "OKP_output.csv"
                      else "OKP_job_" + out.stem[len("OKP_output_"):] + ".txt")
        if not job.is_file():
            continue
        m = re.search(r"(?m)^method:\s*(\S+)", job.read_text(encoding="utf-8"))
        if m and not (data / "kcats" / f"{okp_file(m.group(1))}.tsv").exists() and m.group(1) not in found:
            found.append(m.group(1))
    return found


def _apply_custom_kcats(
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
        from .adapter import resolve_adapter
        adapter = resolve_adapter(
            model,
            purpose="convert_custom_kcats needs a customKcats.tsv path "
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
            "convert_custom_kcats: no reactions were updated. Check that the "
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
    source = format_kcat_source("custom")
    for idx in indices:
        model.ec.kcat[idx] = kcat
        model.ec.source[idx] = source
        if notes:
            existing = model.ec.notes[idx]
            if existing:
                model.ec.notes[idx] = existing + _NOTES_SEPARATOR + notes
            else:
                model.ec.notes[idx] = notes
        updated_indices.add(idx)
