"""Assign kcats from the kcat source files, as model_adapter.toml sets.

``collect_kcats`` decides one kcat per reaction from the source files in
``data/kcats`` without changing the model; ``assign_kcats`` applies the
decisions; ``update_kcat_sources`` gathers the source files the
configuration needs. GECKO (MATLAB) has the same functions as
``collectKcats``, ``assignKcats`` and ``updateKcatSources``.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from ..ec_model.pipeline.apply_custom_kcats import apply_custom_kcats
from ..ec_model.pipeline.apply_kcat import apply_kcat_constraints
from ..ec_model.pipeline.fill_kcats import fill_kcats_from_isozymes
from .get_standard_kcat import assign_standard_kcat
from .kcat_sources import load_kcat_source, read_kcat_source
from .kcats_from import _brenda_release, _measurements, kcats_from_brenda, kcats_from_dlkcat, kcats_from_ec_tree, kcats_from_okp
from .merge_kcats import _strip_okp_prefix, merge_kcats, normalize_source
from .select_kcat_value import apply_kcat_list

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..databases import UniprotDB
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

PRESETS = ("brenda_ec_tree", "ec_tree_okp", "okp", "gecko_median", "gecko_classic")
_DATABASE_TOKENS = {"brenda", "sabio_rk"}
OKP_METHODS = ("CataPro", "CatPred", "CatRange", "DLKcat", "EITLEM", "KinForm-H", "KinForm-L",
               "OmniESI", "OmniESI-O2DENet", "TurNuP", "UniKP")
#: The OKP methods of the ``okp`` source group by default: those with combination
#: parameters (CatRange predicts classes, and has none).
DEFAULT_OKP_METHODS = tuple(m for m in OKP_METHODS if m != "CatRange")


def okp_file(method: str) -> str:
    """The source file name of an OKP method, e.g. ``okp_kinform_h``."""
    return "okp_" + re.sub(r"[^A-Za-z0-9]+", "_", method).lower().strip("_")


@dataclass
class Stage:
    sources: list[str]
    action: str = "select"
    criteria: str = "max"
    top_origin_limit: int = 4
    bottom_origin_limit: int = 6
    wildcard_limit: int = 3


@dataclass
class KcatAssignmentConfig:
    """The kcat assignment settings of an adapter, with a preset expanded."""
    preset: str
    isozymes: bool
    standard: bool
    stages: list[Stage]
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)


def _preset_stages(preset: str) -> tuple[list[Stage], dict[str, dict[str, Any]]]:
    """The presets of kcat assignment; GECKO defines the same."""
    order = ["custom", "database_top", "dlkcat", "database_bottom"]
    own = ["custom", "database_top"]  # with top_origin_limit 1: the organism's own BRENDA value
    brenda = {"brenda": {"aggregation": "median"}}
    if preset == "brenda_ec_tree":
        return ([Stage(sources=own + ["ec_tree"], top_origin_limit=1), Stage(sources=["okp"], action="combine")],
                {**brenda, "ec_tree": {"k": 5}})
    if preset == "ec_tree_okp":
        return ([Stage(sources=own, top_origin_limit=1), Stage(sources=["ec_tree", "okp"], action="combine"),
                 Stage(sources=["okp"], action="combine")], {**brenda, "ec_tree": {"k": 5}})
    if preset == "okp":
        return [Stage(sources=["custom"]), Stage(sources=["okp"], action="combine")], {}
    if preset == "gecko_median":
        return [Stage(sources=order, top_origin_limit=4)], brenda
    if preset == "gecko_classic":
        return [Stage(sources=order, top_origin_limit=6)], {"brenda": {"aggregation": "max"}}
    raise ValueError(f"Unknown kcat assignment preset {preset!r}. Presets: {', '.join(PRESETS)}.")


def kcat_assignment_config(adapter: "ModelAdapter") -> KcatAssignmentConfig:
    """Read ``[kcat_assignment]`` and ``[kcat_sources]`` of the adapter.

    Without a preset or stages the preset is ``brenda_ec_tree``.
    """
    a = adapter.params.kcat_assignment
    if a.stage:
        stages = [Stage(sources=list(s.sources), action=s.action, criteria=s.criteria,
                        top_origin_limit=s.top_origin_limit, bottom_origin_limit=s.bottom_origin_limit,
                        wildcard_limit=s.wildcard_limit) for s in a.stage]
        preset, sources = "", {}
    else:
        preset = a.preset or "brenda_ec_tree"
        stages, sources = _preset_stages(preset)
    sources = {**sources, **{k: dict(v) for k, v in adapter.params.kcat_sources.items()}}
    return KcatAssignmentConfig(preset=preset, isozymes=a.isozymes, standard=a.standard,
                                stages=stages, sources=sources)


def _source_file(token: str) -> str:
    """The source file a stage token reads: the BRENDA tiers read brenda.tsv."""
    return "brenda" if token.startswith("database_") else token


def okp_methods(config: KcatAssignmentConfig) -> list[str]:
    """The OKP methods of the ``okp`` source group (``[kcat_sources.okp] methods``)."""
    methods = config.sources.get("okp", {}).get("methods") or list(DEFAULT_OKP_METHODS)
    unknown = [m for m in methods if m not in OKP_METHODS]
    if unknown:
        raise ValueError(f"Unknown OKP methods in [kcat_sources.okp] methods: {unknown}; "
                         f"methods: {', '.join(OKP_METHODS)}.")
    return list(methods)


def _expand(tokens: list[str], config: KcatAssignmentConfig) -> list[str]:
    """Stage tokens with the ``okp`` group replaced by its methods' files."""
    out: list[str] = []
    for t in tokens:
        out.extend(okp_file(m) for m in okp_methods(config)) if t == "okp" else out.append(t)
    return list(dict.fromkeys(out))


def _combine_params(adapter: "ModelAdapter"):
    from ..databases import fetch_brenda_asset
    from .combine_kcats import load_combine_params

    file = Path(adapter.get_brenda_db_folder()) / "kcatCombineParams.tsv"
    if not file.is_file():
        file = fetch_brenda_asset(["kcatCombineParams.tsv"]) / "kcatCombineParams.tsv"
    return load_combine_params(file)


def _adapter_of(model: "EcModel", adapter: Optional["ModelAdapter"]) -> "ModelAdapter":
    from ..adapter import resolve_adapter

    return resolve_adapter(model, adapter, purpose="kcat assignment needs the adapter")


def _decide(model: "EcModel", adapter: "ModelAdapter", uniprot_db: Optional["UniprotDB"]):
    """Assign kcats to ``model`` in place; return the decisions and the report."""
    config = kcat_assignment_config(adapter)
    path = Path(adapter.params.path)
    kcat_dir = path / "data" / "kcats"
    ec = model.ec
    n = len(ec.rxns)
    ec.kcat = np.zeros(n, dtype=float)
    ec.source = [""] * n
    stage = [""] * n
    candidates = np.zeros(n, dtype=int)
    spread = np.full(n, np.nan)
    report: dict[str, Any] = {"preset": config.preset, "missing": [], "sources": {}}
    loaded: dict[str, Optional[pd.DataFrame]] = {}
    legacy_custom = False

    sd = np.full(n, np.nan)
    params = None

    def load(name: str) -> Optional[pd.DataFrame]:
        nonlocal legacy_custom
        if name not in loaded:
            file = kcat_dir / f"{name}.tsv"
            if file.is_file():
                loaded[name], report["sources"][name] = load_kcat_source(model, file)
            else:
                loaded[name] = None
                if name == "custom" and (path / "data" / "customKcats.tsv").is_file():
                    legacy_custom = True
                else:
                    report["missing"].append(name)
                    logger.warning("No kcat source file %s; it is left out. update_kcat_sources gathers it.", file)
        return loaded[name]

    for s, st in enumerate(config.stages, start=1):
        if st.action == "combine":
            if params is None:
                params = _combine_params(adapter)
            from .combine_kcats import combine_kcats

            lists = {name: load(name) for name in _expand(st.sources, config)}
            index = {r: i for i, r in enumerate(ec.rxns)}
            open_rows = {r for r, k in zip(ec.rxns, ec.kcat) if k == 0}
            lists = {k: v[v["rxn_id"].isin(open_rows)] for k, v in lists.items() if v is not None}
            combined = combine_kcats(lists, params)
            for rxn, kcat, sdv, nsrc, spr in zip(combined["rxn_id"], combined["kcat"], combined["sd"],
                                                 combined["n"], combined["spread_log10"]):
                i = index[rxn]
                ec.kcat[i] = kcat
                ec.source[i] = f"combined (sd={sdv:.2f}, n={nsrc})"
                stage[i], candidates[i], spread[i], sd[i] = str(s), nsrc, spr, sdv
            continue
        lists: list[pd.DataFrame] = []
        used: set[str] = set()
        priority: list[str] = []
        for token in _expand(st.sources, config):
            name = _source_file(token)
            kl = load(name)
            if kl is None or kl.empty:
                continue
            if name not in used:
                lists.append(kl)
                used.add(name)
            if token.startswith("database_"):
                priority.append(token)
            else:
                # The labels of the file's rows, at the file's place in the order.
                labels = dict.fromkeys(_strip_okp_prefix(normalize_source(x)) for x in kl["source"])
                priority.extend(x for x in labels if x not in _DATABASE_TOKENS)
        if not lists or not priority:
            continue
        merged = merge_kcats(*lists, source_priority=priority, top_origin_limit=st.top_origin_limit,
                             bottom_origin_limit=st.bottom_origin_limit, wildcard_limit=st.wildcard_limit)
        if merged.empty:
            continue
        open_before = ec.kcat == 0
        updated = apply_kcat_list(model, merged, criteria=st.criteria, overwrite=False)
        index = {r: i for i, r in enumerate(ec.rxns)}
        valid = merged[merged["kcat"] > 0]
        groups = valid.groupby("rxn_id")["kcat"]
        for rxn in updated:
            i = index[rxn]
            if not open_before[i]:
                continue
            stage[i] = str(s)
            values = groups.get_group(rxn).to_numpy(dtype=float)
            candidates[i] = len(values)
            spread[i] = np.log10(values.max()) - np.log10(values.min())

    if legacy_custom:
        before = ec.kcat.copy()
        apply_custom_kcats(model, path=path / "data" / "customKcats.tsv")
        changed = ec.kcat != before
        for i in np.flatnonzero(changed):
            stage[i], candidates[i], spread[i] = "custom", 1, 0.0
    if config.isozymes and not ec.gecko_light:
        before = ec.kcat == 0
        fill_kcats_from_isozymes(model)
        for i in np.flatnonzero(before & (ec.kcat > 0)):
            stage[i] = "isozymes"
    if config.standard:
        if uniprot_db is None:
            from ..databases import load_uniprot_tsv
            uniprot_db = load_uniprot_tsv(path / "data" / "uniprot.tsv")
        assign_standard_kcat(model, uniprot_db)
        extra = len(ec.rxns) - len(stage)
        stage += [""] * extra
        candidates = np.concatenate([candidates, np.zeros(extra, dtype=int)])
        spread = np.concatenate([spread, np.full(extra, np.nan)])
        sd = np.concatenate([sd, np.full(extra, np.nan)])
        for i, src in enumerate(ec.source):
            if src == "standard" and ec.kcat[i] > 0:
                stage[i] = "standard"

    decisions = pd.DataFrame({
        "ec_rxn": list(ec.rxns), "stage": stage, "source": list(ec.source),
        "kcat": np.asarray(ec.kcat, dtype=float), "candidates": candidates, "spread_log10": spread,
        "sd_log10": sd,
    })
    logger.info("Kcats decided for %d of %d reactions.", int((decisions["stage"] != "").sum()), len(decisions))
    return decisions, report


def _write_decisions(model: "EcModel", adapter: "ModelAdapter", decisions: pd.DataFrame) -> None:
    out = Path(adapter.params.path) / "output"
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out / f"{model.id or 'ecModel'}_kcatDecisions.tsv", sep="\t", index=False,
                     lineterminator="\n")


def collect_kcats(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    uniprot_db: Optional["UniprotDB"] = None,
    write_report: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], "EcModel"]:
    """Decide one kcat per reaction from the kcat source files.

    Combines the kcat source files in ``data/kcats`` as ``[kcat_assignment]``
    in model_adapter.toml sets, without changing the model given;
    :func:`assign_kcats` applies the decisions.

    The order is a preset or ``[[kcat_assignment.stage]]`` tables. A stage
    decides the reactions that are still open:

    - ``select``: the first listed source with a value (see
      :func:`merge_kcats`). Sources are source file names and the BRENDA
      tiers ``database_exact``, ``database_top`` and ``database_bottom``,
      which read ``brenda.tsv``.
    - ``combine``: all listed sources with a value, each corrected for its
      bias and weighted by its reliability for that reaction (see
      :mod:`geckopy.gather_kcats.combine_kcats`); ``ec.source`` reads
      ``combined (sd=<log10 sd>, n=<sources>)``.

    In both, ``okp`` stands for the OKP methods of ``[kcat_sources.okp]
    methods`` (default: all methods with combination parameters).

    The presets:

    - ``brenda_ec_tree`` (the default): custom kcats, the BRENDA median of
      measurements of the organism on the reaction's substrate (origin 1,
      without EC wildcards), the EC tree (:func:`kcats_from_ec_tree`); then
      the OKP methods combined for the reactions left (without an EC number).
    - ``ec_tree_okp``: custom kcats, the organism's own BRENDA value; then
      the EC tree and the OKP methods combined; then the OKP methods
      combined.
    - ``okp``: custom kcats; then the OKP methods combined.
    - ``gecko_median``: custom kcats, the BRENDA median of matches without
      EC wildcards from the organism or other organisms, with or without the
      substrate (origin 1 to 4), DLKcat, then the other BRENDA matches (EC
      wildcards, and specific activities, origin 5 and 6).
    - ``gecko_classic``: as gecko_median with the BRENDA maximum and the
      specific-activity matches above DLKcat (origin up to 6), as the GECKO
      3 tutorial.

    Against measured kcats hidden from the search, the EC tree was closer
    than the BRENDA median at origins 2 and 4 and with EC wildcards and as
    close at origins 1 and 3; the EC tree combined with the OKP methods was
    slightly closer than the EC tree, and the OKP methods combined as close.
    In ecModels, values combined with predictions set high-turnover enzymes
    such as ATP synthase too low, which is why the default uses predictions
    only where there is no value from measurements. The BRENDA aggregation is
    ``[kcat_sources.brenda] aggregation`` (see :func:`update_kcat_sources`).

    After the stages: custom kcats of ``data/customKcats.tsv`` when a stage
    lists custom but there is no ``data/kcats/custom.tsv``; with
    ``isozymes``, kcats from isozymes; with ``standard``, the standard kcat
    for the rest. A source file a stage names that does not exist is
    reported and left out.

    Parameters
    ----------
    model
        The ecModel.
    adapter
        Model adapter (default: ``model.adapter``).
    uniprot_db
        UniProt data for the standard kcat (default: ``data/uniprot.tsv``).
    write_report
        Write the decisions to ``output/<model id>_kcatDecisions.tsv``.

    Returns
    -------
    (decisions, report, ec_model)
        ``decisions`` has one row per ``model.ec`` reaction: ``ec_rxn``,
        ``stage`` (the stage number, ``custom``, ``isozymes``, ``standard``,
        or empty without a kcat), ``source``, ``kcat``, ``candidates`` (values
        the deciding source had) and ``spread_log10``. ``report`` has the
        ``preset``, the ``missing`` source files and, per source file, its
        load report. ``ec_model`` is a copy of the model with the decided
        kcats, before ``apply_kcat_constraints``.
    """
    adapter = _adapter_of(model, adapter)
    ec_model = model.copy()
    decisions, report = _decide(ec_model, adapter, uniprot_db)
    if write_report:
        _write_decisions(model, adapter, decisions)
    return decisions, report, ec_model


def assign_kcats(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    uniprot_db: Optional["UniprotDB"] = None,
    write_report: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign kcats from the kcat source files and apply them, in place.

    Decides one kcat per reaction as :func:`collect_kcats` does, writes
    ``model.ec.kcat`` and ``model.ec.source``, and applies the kcat
    constraints. Any kcats the model had are replaced: the source files are
    the input, and nothing in the model is written back to them.

    Returns
    -------
    (decisions, report)
        As returned by :func:`collect_kcats`.
    """
    adapter = _adapter_of(model, adapter)
    decisions, report = _decide(model, adapter, uniprot_db)
    if write_report:
        _write_decisions(model, adapter, decisions)
    apply_kcat_constraints(model)
    return decisions, report


def _okp_method(name: str, wanted: dict[str, Any]) -> str:
    if "method" in wanted:
        return str(wanted["method"])
    for m in OKP_METHODS:
        if "okp_" + re.sub(r"[^A-Za-z0-9]+", "_", m).lower() == name:
            return m
    raise ValueError(f'Set the OKP method of {name} in model_adapter.toml: [kcat_sources.{name}] method = "..."')


def _same(a: Any, b: Any) -> bool:
    norm = (lambda v: v.lower() if isinstance(v, str) else (int(v) if isinstance(v, bool) else v))
    return norm(a) == norm(b)


def update_kcat_sources(
    model: "EcModel",
    *,
    dry_run: bool = False,
    adapter: Optional["ModelAdapter"] = None,
) -> pd.DataFrame:
    """Gather the kcat sources that the configuration needs.

    Lists the source files that ``[kcat_assignment]`` uses and gathers those
    that are missing or outdated with :func:`kcats_from_brenda`,
    :func:`kcats_from_ec_tree`, :func:`kcats_from_dlkcat` and
    :func:`kcats_from_okp`. A file is outdated when its header records other
    settings than ``[kcat_sources.<name>]`` asks for (presets set these for
    brenda and ec_tree) or, for brenda and ec_tree, another BRENDA release
    than the BRENDA files in use. DLKcat reads
    ``data/DLKcat.tsv`` if it exists and runs DLKcat (Docker) otherwise; OKP
    submits or fetches a job and reports it as pending while it runs.
    ``custom.tsv`` and files it cannot gather are left as they are. Calling
    it again is safe.

    Returns
    -------
    pandas.DataFrame
        One row per source: ``source``, ``status`` (current, gathered,
        pending, outdated, missing, failed or skipped) and ``note``.
    """
    adapter = _adapter_of(model, adapter)
    config = kcat_assignment_config(adapter)
    kcat_dir = Path(adapter.params.path) / "data" / "kcats"
    names = list(dict.fromkeys(_source_file(t) for st in config.stages for t in _expand(st.sources, config)))
    rows = []
    for name in names:
        wanted = config.sources.get(name, {})
        status, note = "current", ""
        file = kcat_dir / f"{name}.tsv"
        if name == "custom":
            status, note = "skipped", "curated by hand"
        elif name not in ("brenda", "dlkcat", "ec_tree") and not name.startswith("okp_"):
            status, note = "skipped", "not a source geckopy gathers"
        elif not file.is_file():
            status = "missing"
        else:
            _, meta = read_kcat_source(file)
            recorded = json.loads(meta.get("parameters") or "{}")
            for key, value in wanted.items():
                if key not in recorded or not _same(recorded[key], value):
                    status, note = "outdated", f"{key} differs from model_adapter.toml"
                    break
            if status == "current" and name == "ec_tree":
                release = _measurements(adapter).attrs.get("release", "")
                if release and meta.get("data") and meta["data"] != release:
                    status, note = "outdated", f"made from {meta['data']}; geckopy has {release}"
            if status == "current" and name == "brenda":
                release = _brenda_release(Path(adapter.get_brenda_db_folder()))
                if release and meta.get("data") and meta["data"] != release:
                    status, note = "outdated", f"made from {meta['data']}; geckopy has {release}"
        if status in ("missing", "outdated") and not dry_run:
            status, note = _gather(model, name, wanted, adapter)
        rows.append({"source": name, "status": status, "note": note})
    report = pd.DataFrame(rows, columns=["source", "status", "note"])
    logger.info("Kcat sources:\n%s", report.to_string(index=False))
    return report


def _gather(model: "EcModel", name: str, wanted: dict[str, Any], adapter: "ModelAdapter") -> tuple[str, str]:
    try:
        if name == "brenda":
            kcats_from_brenda(model, aggregation=wanted.get("aggregation"),
                              force_wildcard_level=int(wanted.get("force_wildcard_level", 0)), adapter=adapter)
        elif name == "ec_tree":
            kcats_from_ec_tree(model, k=float(wanted.get("k", 5)), adapter=adapter)
        elif name == "dlkcat":
            run = not (Path(adapter.params.path) / "data" / "DLKcat.tsv").is_file()
            if run and shutil.which("docker") is None:
                return "failed", "DLKcat needs Docker to run"
            kcats_from_dlkcat(model, run=run, adapter=adapter)
        else:
            method = _okp_method(name, wanted)
            _, done = kcats_from_okp(model, method=method, save_as=name, adapter=adapter)
            if not done:
                return "pending", f"OKP job for {method} still running; call again later"
    except Exception as err:  # reported per source, as GECKO does
        return "failed", str(err)
    return "gathered", ""
