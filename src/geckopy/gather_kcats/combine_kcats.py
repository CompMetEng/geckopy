"""Combine several kcat sources into one value per reaction.

Each source's value for a reaction is corrected for the source's bias and
weighted by 1 / its error variance; both depend on the source and its
evidence for that reaction (the evidence bin):

- ``ec_tree``: the number of measurements of the full EC number, or the
  deepest EC level with measurements (``ec4:1-2`` ... ``ec4:>100``, ``ec3``,
  ``ec2``, ``ec0``);
- ``homologs``: the identity of the nearest homolog;
- ``okp_<method>``: the enzyme's similarity to the method's training
  sequences, in percent (``<40``, ``40-60``, ``60-80``, ``>80``).

The combined value is the precision-weighted mean of the corrected values,
and its standard deviation for a new measurement is
``sqrt(1 / sum(1 / variance) + measurement_sd**2)``, scaled so that the 80%
interval covers 80% of the measurements the parameters were fitted on. The
parameters (``kcatCombineParams.tsv``) were fitted once on the kcat
benchmark's development split and come with the BRENDA files. GECKO does the
same in ``collectKcats``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CombineParams:
    """Bias and error variance per source and evidence bin."""
    table: pd.DataFrame  # source, bin, bias, variance, n
    measurement_sd: float
    min_variance: float
    scales: dict[str, float]

    def sources(self) -> set[str]:
        return set(self.table["source"])


def load_combine_params(path: str | Path) -> CombineParams:
    """Read ``kcatCombineParams.tsv``."""
    path = Path(path)
    meta: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    k = 0
    while k < len(lines) and lines[k].startswith("#"):
        m = re.match(r"^#\s*([^:]+):\s*(.*)$", lines[k])
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
        k += 1
    table = pd.read_csv(path, sep="\t", comment="#", dtype={"source": str, "bin": str})
    scales = {key[len("interval_scale "):]: float(v) for key, v in meta.items() if key.startswith("interval_scale ")}
    return CombineParams(table=table, measurement_sd=float(meta.get("measurement_sd", 0.48)),
                         min_variance=float(meta.get("min_variance", 0.01)), scales=scales)


def evidence_bin(source: str, detail: str) -> str:
    """The evidence bin of a source's value from its detail column."""
    detail = detail or ""
    if source == "ec_tree":
        n = re.search(r"n=(\d+)", detail)
        depth = re.search(r"depth=(\d)", detail)
        if depth and depth.group(1) == "4" and n:
            v = int(n.group(1))
            for hi, label in ((2, "1-2"), (5, "3-5"), (20, "6-20"), (100, "21-100")):
                if v <= hi:
                    return f"ec4:{label}"
            return "ec4:>100"
        return f"ec{depth.group(1) if depth else '0'}"
    if source == "homologs":
        m = re.search(r"identity=([\d.]+)", detail)
        if not m:
            return ""
        v = float(m.group(1))
        for hi, label in ((0.5, "40"), (0.6, "50"), (0.7, "60"), (0.8, "70"), (0.9, "80")):
            if v <= hi:
                return label
        return "90"
    if source.startswith("okp_"):
        m = re.search(r"similarity=([\d.]+)", detail)
        if not m:
            return ""
        v = float(m.group(1))
        if v <= 40:
            return "<40"
        if v <= 60:
            return "40-60"
        if v <= 80:
            return "60-80"
        return ">80"
    return ""


def _scale_key(sources: set[str]) -> str:
    has_okp = any(s.startswith("okp_") for s in sources)
    if "ec_tree" in sources and has_okp:
        return "ec_tree+okp"
    if has_okp:
        return "okp"
    if "ec_tree" in sources:
        return "ec_tree+homologs"
    return ""


def combine_kcats(lists: dict[str, pd.DataFrame], params: CombineParams) -> pd.DataFrame:
    """One combined kcat per reaction from the kcat lists of several sources.

    Parameters
    ----------
    lists
        Per source name (``ec_tree``, ``homologs``, ``okp_<method>``) its
        kcat list as :func:`load_kcat_source` returns it (``rxn_id``,
        ``kcat``, ``detail``). Several rows of one source for a reaction
        (substrates, isozymes) give that source's median, with the evidence
        bin of its row of best evidence.
    params
        From :func:`load_combine_params`.

    Returns
    -------
    pandas.DataFrame
        ``rxn_id``, ``kcat``, ``sd`` (log10), ``n`` (sources),
        ``spread_log10`` (range of the sources' values) and ``sources``.
    """
    unknown = sorted(set(lists) - params.sources())
    if unknown:
        raise ValueError(f"No combination parameters for {unknown}; known sources: "
                         f"{sorted(params.sources())}.")
    table = params.table.set_index(["source", "bin"])
    by_source = {s: g for s, g in params.table.groupby("source")}
    parts = []
    for source, kl in lists.items():
        if kl is None or kl.empty:
            continue
        df = pd.DataFrame({"rxn_id": kl["rxn_id"], "log10": np.log10(kl["kcat"].astype(float)),
                           "detail": kl["detail"] if "detail" in kl else ""})
        df = df[np.isfinite(df["log10"])]
        df["bin"] = [evidence_bin(source, d) for d in df["detail"]]
        rank = {b: i for i, b in enumerate(by_source[source].sort_values("variance")["bin"])}
        rows = []
        for rxn, g in df.groupby("rxn_id", sort=False):
            known = [b for b in g["bin"] if (source, b) in table.index]
            # The bin with the smallest error variance among the source's rows; a value
            # without a known bin gets the source's largest variance.
            if known:
                b = min(known, key=lambda x: rank[x])
                bias, var = table.loc[(source, b), ["bias", "variance"]]
            else:
                worst = by_source[source].sort_values("variance").iloc[-1]
                bias, var = worst["bias"], worst["variance"]
            rows.append((rxn, source, float(np.median(g["log10"])), float(bias), max(float(var), params.min_variance)))
        parts.append(pd.DataFrame(rows, columns=["rxn_id", "source", "log10", "bias", "variance"]))
    if not parts:
        return pd.DataFrame(columns=["rxn_id", "kcat", "sd", "n", "spread_log10", "sources"])
    f = pd.concat(parts, ignore_index=True)
    f["x"] = f["log10"] - f["bias"]
    f["w"] = 1.0 / f["variance"]
    out = []
    for rxn, g in f.groupby("rxn_id", sort=False):
        w = g["w"].to_numpy()
        mu = float(np.sum(w * g["x"]) / np.sum(w))
        sources = set(g["source"])
        scale = params.scales.get(_scale_key(sources), 1.0)
        sd = math.sqrt(1.0 / float(np.sum(w)) + params.measurement_sd ** 2) * scale
        out.append((rxn, 10 ** mu, sd, len(g), float(g["log10"].max() - g["log10"].min()),
                    ", ".join(sorted(sources))))
    return pd.DataFrame(out, columns=["rxn_id", "kcat", "sd", "n", "spread_log10", "sources"])
