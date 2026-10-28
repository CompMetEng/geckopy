"""Kcat source files: gathered kcats kept in the project, one file per source.

A kcat source file, ``data/kcats/<name>.tsv`` in the project folder, holds
the kcats of one source (BRENDA, DLKcat, an OpenKineticsPredictor method,
or curated values). Rows are keyed by their conventional reaction and
enzyme set rather than by ``model.ec.rxns``, so a file stays valid when the
ecModel is rebuilt, and applies to full and light models alike.

The file is UTF-8 text with LF line endings. It starts with ``#`` header
lines (source, generator, parameters, data, kind, created), then a
tab-separated table with the columns ``rxn``, ``enzymes``, ``substrate``,
``kcat``, ``log10_sd``, ``source``, ``detail``, ``ec_code`` and ``note``,
sorted by reaction, enzymes, substrate and source. GECKO (MATLAB) reads and
writes the same files with ``saveKcatSource`` and ``loadKcatSource``.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

COLUMNS = ["rxn", "enzymes", "substrate", "kcat", "log10_sd", "source", "detail", "ec_code", "note"]
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_EXP_RE = re.compile(r"_EXP_\d+$")
_LIGHT_RE = re.compile(r"^\d+_")
_NUMBER_RE = re.compile(r"^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$|^[-+]?Inf$")
_META_RE = re.compile(r"^\s*#\s*([A-Za-z][A-Za-z0-9_]{0,62})\s*:\s?(.*)$", re.ASCII)


def ec_rxn_keys(model: "EcModel") -> tuple[list[str], list[str]]:
    """Build-independent keys of the ``model.ec`` reactions.

    Returns
    -------
    (base_rxns, enzyme_sets)
        Per ``model.ec.rxns`` entry: the conventional reaction (without a
        trailing ``_EXP_<n>`` in a full model, or a leading ``<nnn>_`` in a
        light model; a reverse direction keeps its ``_REV``), and the UniProt
        accessions of its enzymes, sorted and joined by ``" + "`` (empty
        without enzymes).
    """
    ec = model.ec
    pattern = _LIGHT_RE if ec.gecko_light else _EXP_RE
    base = [pattern.sub("", r) for r in ec.rxns]
    mat = ec.rxn_enz_mat.tocsr()
    enzymes = []
    for i in range(len(ec.rxns)):
        if mat.shape[1]:
            row = mat[i]
            cols = row.indices[row.data != 0]
        else:
            cols = []
        enzymes.append(" + ".join(sorted({ec.enzymes[j] for j in cols})))
    return base, enzymes


def format_number(x: Any) -> str:
    """Whole numbers as integers, others in the shortest decimal form that
    reads back as the same float; ``"Inf"`` for infinity, ``""`` for NaN.
    Matches GECKO's ``saveKcatSource``."""
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == round(x) and abs(x) < 1e15:
        return str(int(round(x)))
    for digits in range(1, 18):
        text = f"{x:.{digits}g}"
        if float(text) == x:
            return text
    return repr(x)


def _json_string(text: str) -> str:
    out = []
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch in "\b\t\n\f\r":
            out.append({"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}[ch])
        elif ord(ch) < 32:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _parameters_json(value: Any) -> str:
    """The parameters line as compact JSON with keys sorted at every level.

    Numbers are written with :func:`format_number`, NaN and infinity as
    ``null``; text as it is, non-ASCII included. GECKO's ``saveKcatSource``
    writes the same text.
    """
    if value is None:
        return "{}"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_json_string(str(k))}:{_parameters_json_value(value[k])}"
                              for k in sorted(value, key=str)) + "}"
    return _parameters_json_value(value)


def _parameters_json_value(value: Any) -> str:
    if isinstance(value, dict):
        return _parameters_json(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_parameters_json_value(v) for v in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "null" if not math.isfinite(float(value)) else format_number(value)
    if isinstance(value, str):
        return _json_string(value)
    raise TypeError(f"Parameters of type {type(value).__name__} cannot be written.")


def _project_path(model: "EcModel", adapter: Optional["ModelAdapter"]) -> Path:
    from ..adapter import resolve_adapter

    adapter = resolve_adapter(model, adapter, purpose="kcat source files need params.path")
    return Path(adapter.params.path)


def _header_line(text: str) -> Optional[tuple[str, str]]:
    """Key (in lower case) and value of a ``# key: value`` header line."""
    m = _META_RE.match(_unquote(text))
    return (m.group(1).lower(), m.group(2).strip()) if m else None


def _read_header_lines(path: Path) -> dict[str, str]:
    """Only the ``#`` lines at the top."""
    meta: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for line in f:
            text = line.rstrip("\r\n")
            if not text.strip():
                continue
            if not _unquote(text).startswith("#"):
                break
            kv = _header_line(text)
            if kv:
                meta[kv[0]] = kv[1]
    return meta


def kcat_source_file(
    name: str,
    *,
    model: Optional["EcModel"] = None,
    adapter: Optional["ModelAdapter"] = None,
    folder: Optional[str | Path] = None,
) -> Path:
    """Path of a kcat source file that may be written.

    Checks the name, resolves the folder (``data/kcats`` in the project
    folder by default), and refuses a file curated by hand: named
    ``custom.tsv``, or whose header says ``kind: curated`` or
    ``source: custom`` (in any case).
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"The source name {name!r} may only contain letters, digits, '_', '-' and '.'.")
    folder = Path(folder) if folder is not None else _project_path(model, adapter) / "data" / "kcats"
    path = folder / f"{name}.tsv"
    if name.lower() == "custom":
        raise PermissionError("custom.tsv holds curated kcats and is not written by geckopy; edit it by hand.")
    if path.is_file():
        meta = _read_header_lines(path)
        if meta.get("kind", "").lower() == "curated" or meta.get("source", "").lower() == "custom":
            raise PermissionError(f"{path} is curated by hand and is not overwritten.")
    return path


def _geckopy_version() -> str:
    try:
        return version("geckopy")
    except PackageNotFoundError:
        return ""


def _clean(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = "; ".join(str(v) for v in value)
    elif value is None or pd.isna(value):
        return ""
    return re.sub(r"[\t\r\n]", " ", str(value))


def _detail_text(wildcard_level, origin, extra) -> str:
    """The detail column: ``wc=``, ``origin=`` and the free text, joined by ``; ``.

    Tabs and line breaks of the free text become spaces, so a row can never be split. Empty
    parts leave no ``; `` behind: a leading or trailing ``; `` is dropped and a repeated
    ``; `` is written once, as ``saveKcatSource`` does.
    """
    parts = [
        f"wc={int(wildcard_level)}" if wildcard_level is not None and pd.notna(wildcard_level) else "",
        f"origin={int(origin)}" if origin is not None and pd.notna(origin) else "",
        _clean(extra).strip(),
    ]
    text = "; ".join(parts)
    text = re.sub(r"^(; )+|(; )+$", "", text)
    return re.sub(r"(; ){2,}", "; ", text)


def save_kcat_source(
    model: "EcModel",
    kcat_list: pd.DataFrame,
    name: str,
    *,
    adapter: Optional["ModelAdapter"] = None,
    folder: Optional[str | Path] = None,
    generator: str = "save_kcat_source",
    parameters: Optional[dict] = None,
    data: str = "",
    kind: str = "gathered",
) -> Path:
    """Write a kcat list to the kcat source file ``<name>.tsv``.

    Left out are rows with kcat 0, NaN or below (no value found), and rows
    of ``model.ec`` reactions without enzymes, which have no enzyme
    constraint and could not be told apart from their isozymes.

    Parameters
    ----------
    model
        The ecModel the kcat list refers to.
    kcat_list
        A kcat list as returned by the ``kcats_from_*`` functions,
        ``fuzzy_kcat_matching``, ``read_dlkcat_output`` or ``merge_kcats``:
        columns ``rxn_id`` and ``kcat``, optionally ``substrates``,
        ``eccode``, ``wildcard_level``, ``origin``, ``source``,
        ``log10_sd`` and ``detail`` (further ``key=value`` pairs for the
        detail column).
    name
        Name of the source; the file is ``<name>.tsv``.
    adapter
        Model adapter (default: ``model.adapter``); gives the project folder.
    folder
        Folder to write to (default: ``data/kcats`` in the project folder).
    generator, parameters, data
        What produced the values, its settings (written as JSON with sorted
        keys) and the version of the data they come from.
    kind
        ``"gathered"`` (default) or ``"curated"``.

    A file named ``custom.tsv``, or whose header says ``kind: curated`` or
    ``source: custom``, is never written.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    if kind not in ("gathered", "curated"):
        raise ValueError("kind must be 'gathered' or 'curated'")
    path = kcat_source_file(name, model=model, adapter=adapter, folder=folder)

    base, enzymes = ec_rxn_keys(model)
    index = {r: i for i, r in enumerate(model.ec.rxns)}
    unknown = sorted({r for r in kcat_list["rxn_id"] if r not in index})
    if unknown:
        raise ValueError(f"These kcat list reactions are not in model.ec.rxns: {unknown}")

    def col(column):
        if column in kcat_list.columns:
            return kcat_list[column]
        default = name if column == "source" else None
        return pd.Series([default] * len(kcat_list), index=kcat_list.index)

    rows = []
    no_enzymes = 0
    for rxn, kcat, subs, src, wc, origin, ec_code, sd, extra in zip(
        kcat_list["rxn_id"], kcat_list["kcat"], col("substrates"), col("source"),
        col("wildcard_level"), col("origin"), col("eccode"), col("log10_sd"), col("detail"),
    ):
        if kcat is None or pd.isna(kcat) or not (math.isfinite(float(kcat)) and float(kcat) > 0):
            continue
        i = index[rxn]
        if not enzymes[i]:
            no_enzymes += 1
            continue
        rows.append([
            _clean(base[i]), _clean(enzymes[i]), _clean(subs), format_number(kcat),
            format_number(sd), _clean(src), _detail_text(wc, origin, extra), _clean(ec_code), "",
        ])
    if no_enzymes:
        logger.info("%d kcats of reactions without enzymes are not written to %s.", no_enzymes, path)
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[5], r[3], r[4], r[6], r[7], r[8]))

    path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = [
        f"# source: {name}",
        f"# generator: {_clean(generator)} (geckopy {_geckopy_version()})",
        f"# parameters: {_parameters_json(parameters)}",
        f"# data: {_clean(data)}",
        f"# kind: {kind}",
        f"# created: {created}",
        "\t".join(COLUMNS),
    ]
    path.write_text("\n".join(header + ["\t".join(r) for r in rows]) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def _unquote(text: str) -> str:
    """A field without surrounding spaces and double quotes; a field wrapped
    in double quotes, as a spreadsheet may save it, is unwrapped with ``""``
    for a quote inside, and spaces inside the quotes are trimmed too."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].replace('""', '"').strip()
    return text


def _parse_number(text: str) -> float:
    return float(text) if _NUMBER_RE.match(text) else math.nan


def read_kcat_source(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a kcat source file as it is, without a model.

    The file is read as UTF-8 (a leading byte-order mark is ignored). The
    ``#`` lines before the column header are the header; ``#`` lines and
    blank lines further down are skipped. Fields wrapped in double quotes,
    as a spreadsheet may save them, are unwrapped. ``kcat`` and
    ``log10_sd`` must be plain numbers (``12``, ``0.5``, ``1e-05``); other
    text, such as ``5,5`` with a decimal comma, is read as NaN, which
    :func:`load_kcat_source` reports. A row with more filled fields than
    there are columns is an error.

    Returns
    -------
    (records, meta)
        ``records`` has one row per data row: ``kcat`` and ``log10_sd`` as
        floats (NaN when empty or not a number), every other column as text;
        columns other than the standard ones are kept, and ``line`` is the
        line number of each row in the file. ``meta`` holds the ``#`` header
        lines as ``{key: value}``.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        lines = re.split(r"\r?\n", f.read())
    meta: dict[str, str] = {}
    k = 0
    while k < len(lines) and (not lines[k].strip() or _unquote(lines[k]).startswith("#")):
        kv = _header_line(lines[k])
        if kv:
            meta[kv[0]] = kv[1]
        k += 1
    if k >= len(lines):
        raise ValueError(f"{path} has no column header.")
    columns = [_unquote(c) for c in lines[k].split("\t")]
    if not {"kcat", "source"} <= set(columns) or not {"rxn", "enzymes"} & set(columns):
        raise ValueError(f"{path} needs the columns kcat and source, and rxn and/or enzymes.")
    named = [c for c in columns if c]
    if len(set(named)) < len(named) or "line" in columns:
        raise ValueError(f"{path} has a column name twice, or a column named line.")
    rows, line_numbers = [], []
    for j in range(k + 1, len(lines)):
        line = lines[j]
        if not line.strip() or _unquote(line).startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) > len(columns):
            if any(f.strip() for f in fields[len(columns):]):
                raise ValueError(f"{path}, line {j + 1} has more fields than the {len(columns)} columns.")
            fields = fields[: len(columns)]
        fields += [""] * (len(columns) - len(fields))
        if not any(f.strip() for f in fields):
            continue  # a row of tabs only
        rows.append([_unquote(f) for f in fields])
        line_numbers.append(j + 1)
    keep = [i for i, c in enumerate(columns) if c]
    records = pd.DataFrame([[r[i] for i in keep] for r in rows], columns=[columns[i] for i in keep], dtype=object)
    for c in COLUMNS:
        if c not in records.columns:
            records[c] = pd.Series([""] * len(records), dtype=object)
    for c in ("kcat", "log10_sd"):
        records[c] = pd.Series([_parse_number(v) for v in records[c]], dtype=float)
    records["line"] = pd.Series(line_numbers, dtype=int)
    return records, meta


def _normalize_enzymes(text: str) -> str:
    parts = sorted({p.strip() for p in text.split("+") if p.strip()})
    return " + ".join(parts)


def load_kcat_source(
    model: "EcModel",
    name: str | Path,
    *,
    adapter: Optional["ModelAdapter"] = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read a kcat source file and map its rows to the ``model.ec`` reactions.

    1. A row with ``rxn`` and ``enzymes`` applies to the ``model.ec`` rows
       with that conventional reaction and that enzyme set.
    2. A row with only ``enzymes`` applies to every row with that enzyme
       set; a row with only ``rxn`` to every row of that reaction (curated
       rows; :func:`save_kcat_source` always writes both).
    3. A row whose kcat is missing, not a number, infinite, or not above 0
       is reported as invalid and not applied.
    4. A row that matches nothing is reported, never dropped silently. If a
       ``model.ec`` row has at least half of the row's enzymes, that partial
       match is reported as well, but not applied.

    The enzymes of a row may be listed in any order.

    Parameters
    ----------
    model
        The ecModel.
    name
        Name of the source (``data/kcats/<name>.tsv`` in the project folder),
        or the path to a kcat source file.

    Returns
    -------
    (kcat_list, report)
        ``kcat_list`` has one row per applied match, in the kcat-list schema
        (``rxn_id``, ``source``, ``eccode``, ``substrates``, ``genes``,
        ``kcat``, ``wildcard_level``, ``origin``) plus ``log10_sd``,
        ``detail`` (the detail column) and ``line`` (the line of the file). ``report`` has the ``invalid``,
        ``unmatched`` and ``partial`` rows.
    """
    path = Path(name)
    if not path.is_file():
        path = _project_path(model, adapter) / "data" / "kcats" / f"{name}.tsv"
        if not path.is_file():
            raise FileNotFoundError(
                f"No kcat source file {path}. Make it with the kcats_from_* function of this "
                f"source (kcats_from_brenda, kcats_from_dlkcat, kcats_from_okp), or with "
                f"save_kcat_source."
            )
    records, meta = read_kcat_source(path)
    base, enzymes = ec_rxn_keys(model)

    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_base: dict[str, list[int]] = defaultdict(list)
    by_enz: dict[str, list[int]] = defaultdict(list)
    by_acc: dict[str, list[int]] = defaultdict(list)
    acc_sets = [set(e.split(" + ")) if e else set() for e in enzymes]
    for i, (b, e) in enumerate(zip(base, enzymes)):
        by_pair[(b, e)].append(i)
        by_base[b].append(i)
        if e:
            by_enz[e].append(i)
        for a in acc_sets[i]:
            by_acc[a].append(i)

    kcats = records["kcat"].to_numpy(dtype=float)
    invalid = [r for r in range(len(records)) if not (math.isfinite(kcats[r]) and kcats[r] > 0)]
    invalid_set = set(invalid)
    rec_idx, ec_idx, unmatched, partial = [], [], [], []
    for r, (rxn, enz_text) in enumerate(zip(records["rxn"], records["enzymes"])):
        if r in invalid_set:
            continue
        enz = _normalize_enzymes(enz_text) if enz_text else ""
        if rxn and enz:
            hit = by_pair.get((rxn, enz), [])
        elif enz:
            hit = by_enz.get(enz, [])
        elif rxn:
            hit = by_base.get(rxn, [])
        else:
            hit = []
        if not hit:
            unmatched.append(r)
            if enz:
                wanted = enz.split(" + ")
                candidates = sorted({c for a in wanted for c in by_acc.get(a, [])})
                if rxn:
                    candidates = [c for c in candidates if base[c] == rxn]
                for c in candidates:
                    if np.mean([w in acc_sets[c] for w in wanted]) >= 0.5:
                        partial.append((r, c))
        else:
            rec_idx.extend([r] * len(hit))
            ec_idx.extend(hit)

    sub = records.iloc[rec_idx]
    wc, origin = [], []
    for d in sub["detail"]:
        m = re.search(r"wc=(\d+)", d or "")
        wc.append(float(m.group(1)) if m else np.nan)
        m = re.search(r"origin=(\d+)", d or "")
        origin.append(float(m.group(1)) if m else np.nan)
    kcat_list = pd.DataFrame({
        "rxn_id": [model.ec.rxns[i] for i in ec_idx],
        "source": sub["source"].tolist(),
        "eccode": sub["ec_code"].tolist(),
        "substrates": [[s.strip() for s in t.split("; ") if s.strip()] for t in sub["substrate"]],
        "genes": [[] for _ in rec_idx],
        "kcat": sub["kcat"].to_numpy(dtype=float),
        "wildcard_level": pd.Series(wc, dtype=float),
        "origin": pd.Series(origin, dtype=float),
        "log10_sd": sub["log10_sd"].to_numpy(dtype=float),
        "detail": sub["detail"].tolist(),
        "line": pd.Series(sub["line"].tolist(), dtype=int),
    })
    kcat_list.attrs["source_name"] = meta.get("source", "")

    def rows_of(idx):
        return pd.DataFrame({
            "line": records["line"].iloc[idx].tolist(),
            "rxn": records["rxn"].iloc[idx].tolist(),
            "enzymes": records["enzymes"].iloc[idx].tolist(),
            "kcat": records["kcat"].iloc[idx].tolist(),
        })

    report = {
        "invalid": rows_of(invalid),
        "unmatched": rows_of(unmatched),
        "partial": pd.DataFrame({
            "line": [int(records["line"].iloc[r]) for r, _ in partial],
            "rxn": [records["rxn"].iloc[r] for r, _ in partial],
            "enzymes": [records["enzymes"].iloc[r] for r, _ in partial],
            "ec_rxn": [model.ec.rxns[c] for _, c in partial],
            "ec_enzymes": [enzymes[c] for _, c in partial],
        }),
    }
    if invalid:
        logger.warning("%d of %d rows of %s have no valid kcat; see report['invalid'].",
                       len(invalid), len(records), path)
    if unmatched:
        logger.warning("%d of %d rows of %s apply to no reaction of this model; see report['unmatched'].",
                       len(unmatched), len(records), path)
    return kcat_list, report
