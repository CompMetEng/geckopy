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
import re
import shutil
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .ec_model.constants import POOL_EXCHANGE_ID, POOL_ID

if TYPE_CHECKING:
    from .adapter import ModelAdapter
    from .ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

#: Renamed or replaced functions, as (old name, what to use).
RENAMED = (
    ("get_standard_kcat", "assign_standard_kcat"),
    ("select_kcat_value", "apply_kcat_list"),
    ("merge_dlkcat_and_fuzzy_kcats", "merge_kcats"),
    ("fuzzy_kcat_matching", "kcats_from_brenda (or kcats_from_ec_tree), then assign_kcats"),
    ("read_dlkcat_output", "kcats_from_dlkcat, then assign_kcats"),
    ("parse_okp_output", "kcats_from_okp, then assign_kcats"),
    ("apply_custom_kcats", "data/kcats/custom.tsv and assign_kcats"),
)


def convert_custom_kcats(
    model: "EcModel",
    *,
    custom_kcats: Optional[str | Path] = None,
    out_file: Optional[str | Path] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> tuple[Path, dict]:
    """Convert ``data/customKcats.tsv`` into ``data/kcats/custom.tsv``.

    The kcats are those :func:`~geckopy.ec_model.pipeline.apply_custom_kcats`
    gives the ecModel, including which of two entries for the same reaction
    wins: one row per reaction, keyed by conventional reaction and enzyme set
    (see :func:`~geckopy.gather_kcats.ec_rxn_keys`), with the entry's notes in
    the note column. The ``stoicho`` column is left out: complex
    stoichiometry comes from ``apply_complex_data``. Unlike
    ``apply_custom_kcats``, ``custom.tsv`` also applies to light ecModels.

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
    from .ec_model.pipeline.apply_custom_kcats import apply_custom_kcats
    from .gather_kcats.kcat_sources import _geckopy_version, ec_rxn_keys, format_number

    if model.ec.gecko_light:
        raise ValueError("apply_custom_kcats does not work on light ecModels; give the full ecModel. "
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
    apply_custom_kcats(applied, path=src, apply=False)
    base, enzymes = ec_rxn_keys(applied)
    rows: dict[tuple[str, str], list[str]] = {}
    for i in range(n):
        if applied.ec.source[i] != "custom":
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

        # read_yaml_model already turns GECKO 3 protein reactions forward, with a
        # warning that names how many; the file itself keeps the old direction.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = load_ec_model(path)
        flipped = [re.search(r"for (\d+) protein", str(w.message)) for w in caught]
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
