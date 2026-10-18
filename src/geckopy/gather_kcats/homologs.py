"""Kcats from measured homologs of the model's enzymes.

For each reaction, enzyme and substrate (the substrates DLKcat would use,
see :func:`~geckopy.gather_kcats.extract_enzyme_substrate_pairs`), the
enzyme's sequence is searched with DIAMOND against the sequences of the
BRENDA measurements (at least 40% identity, 80% coverage of both sequences).
Each homolog measured on the same substrate (same InChIKey, without the
protonation flag) contributes the median of its measurements; the estimate
is the mean of the ``k`` most identical such homologs, weighted by identity.
Model metabolites are matched to InChIKeys through ``substrates.tsv`` of the
BRENDA files: their ChEBI annotations, then MetaNetX annotations, then their
names. GECKO has the same in ``kcatsFromHomologs``.
"""
from __future__ import annotations

import html
import logging
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ω": "omega",
          "κ": "kappa", "λ": "lambda", "μ": "mu", "ψ": "psi", "ξ": "xi", "ζ": "zeta"}


def compact_name(name: str) -> str:
    """A metabolite name as the compact keys of ``substrates.tsv``: without
    markup, Greek letters spelled out, accents dropped, lower case, and only
    letters, digits and ``+`` kept."""
    s = html.unescape(re.sub(r"<[^>]+>", "", str(name)))
    for g, w in _GREEK.items():
        s = s.replace(g, w).replace(g.upper(), w)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9+]", "", s)


def _annotations(met, key: str) -> list[str]:
    value = (met.annotation or {}).get(key)
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).split(":")[-1].strip() for v in values if str(v).strip()]


def metabolite_inchikeys(model: "EcModel", substrates: pd.DataFrame) -> dict[str, str]:
    """InChIKey (first 25 characters) per model metabolite id, where known:
    from its ChEBI annotations, then its MetaNetX annotations, then its name."""
    lookup: dict[str, dict[str, str]] = {}
    for kind, value, ik in zip(substrates["kind"], substrates["value"].astype(str), substrates["inchikey"]):
        lookup.setdefault(kind, {}).setdefault(value, ik)  # the first entry of a value
    out = {}
    for met in model.metabolites:
        ik = None
        for kind, key in (("chebi", "chebi"), ("metanetx", "metanetx.chemical")):
            for v in _annotations(met, key):
                ik = lookup.get(kind, {}).get(v)
                if ik:
                    break
            if ik:
                break
        if not ik and met.name:
            ik = lookup.get("name", {}).get(compact_name(met.name))
        if ik:
            out[met.id] = ik
    return out


def _brenda_file(adapter: "ModelAdapter", name: str) -> Path:
    from ..databases import fetch_brenda_asset

    path = Path(adapter.get_brenda_db_folder()) / name
    return path if path.is_file() else fetch_brenda_asset([name]) / name


def _read_fasta(path: Path) -> dict[str, str]:
    seqs, key = {}, None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            key = line[1:].split()[0]
            seqs[key] = ""
        elif key is not None:
            seqs[key] += line.strip()
    return seqs


def diamond_hits(queries: dict[str, str], targets: Path, *, min_identity: float = 0.4, coverage: float = 0.8,
                 threads: Optional[int] = None, diamond: Optional[str] = None) -> pd.DataFrame:
    """DIAMOND hits (``query``, ``target``, ``identity`` as a fraction) of
    the query sequences against a FASTA file, best identity per pair."""
    from raven_toolbox.binaries import resolve_binary

    exe = resolve_binary("diamond", binary=diamond)
    with tempfile.TemporaryDirectory(prefix="geckopy-homologs-") as tmp:
        tmp = Path(tmp)
        with open(tmp / "queries.fasta", "w", encoding="ascii") as f:
            for k, v in queries.items():
                f.write(f">{k}\n{v}\n")
        subprocess.run([exe, "makedb", "--in", str(targets), "--db", str(tmp / "targets"), "--quiet"], check=True)
        cmd = [exe, "blastp", "--query", str(tmp / "queries.fasta"), "--db", str(tmp / "targets"),
               "--out", str(tmp / "hits.tsv"), "--ultra-sensitive", "--id", f"{100 * min_identity:g}",
               "--query-cover", f"{100 * coverage:g}", "--subject-cover", f"{100 * coverage:g}",
               "--evalue", "0.001", "--max-target-seqs", "0", "--outfmt", "6", "qseqid", "sseqid", "pident", "--quiet"]
        if threads:
            cmd += ["--threads", str(threads)]
        subprocess.run(cmd, check=True)
        hits = pd.read_csv(tmp / "hits.tsv", sep="\t", names=["query", "target", "pident"],
                           dtype={"query": str, "target": str})
    hits["identity"] = hits["pident"] / 100
    return hits.groupby(["query", "target"], as_index=False)["identity"].max()


def kcats_from_homologs(
    model: "EcModel",
    *,
    k: int = 10,
    min_identity: float = 0.4,
    coverage: float = 0.8,
    threads: Optional[int] = None,
    diamond: Optional[str] = None,
    save: bool = True,
    save_as: str = "homologs",
    adapter: Optional["ModelAdapter"] = None,
) -> pd.DataFrame:
    """Estimate kcats from measured homologs and keep them as ``homologs.tsv``.

    See :mod:`geckopy.gather_kcats.homologs`. One row per reaction and
    substrate; for an enzyme complex the median of its subunits' estimates.
    Reactions without a homolog measured on one of their substrates get no
    value.

    Parameters
    ----------
    model
        EcModel with enzyme sequences (``ec.sequence``).
    k
        The number of most identical homologs averaged.
    min_identity, coverage
        DIAMOND thresholds: identity, and coverage of both sequences.
    threads, diamond
        DIAMOND threads, and its executable (default: found by raven-toolbox,
        which downloads it when needed).
    save, save_as
        Write the source file, and its name without ``.tsv``.
    adapter
        Model adapter (default: ``model.adapter``).

    Returns
    -------
    pandas.DataFrame
        The kcat list; ``detail`` has the identity of the nearest homolog
        (``identity``) and the number of homologs averaged (``n``).
    """
    from ..adapter import resolve_adapter
    from ..databases import load_brenda_measurements, load_dlkcat_ignore_lists
    from .kcat_sources import save_kcat_source
    from .write_dlkcat_input import extract_enzyme_substrate_pairs

    adapter = resolve_adapter(model, adapter, purpose="kcats_from_homologs needs the BRENDA folder and project folder")
    substrates = pd.read_csv(_brenda_file(adapter, "substrates.tsv"), sep="\t", comment="#", dtype=str)
    measurements = load_brenda_measurements(_brenda_file(adapter, "measurements.tsv").parent)
    sequences_file = _brenda_file(adapter, "sequences.fasta")

    pairs = extract_enzyme_substrate_pairs(model, load_dlkcat_ignore_lists(Path(adapter.params.path) / "data"),
                                           only_with_smiles=False)
    iks = metabolite_inchikeys(model, substrates)
    pairs = pairs.assign(inchikey=pairs["met_id"].map(iks))
    pairs = pairs[pairs["inchikey"].notna() & (pairs["sequence"].astype(str) != "")]
    columns = ["rxn_id", "source", "substrates", "kcat", "detail"]
    if pairs.empty:
        kcat_list = pd.DataFrame(columns=columns)
    else:
        seq_ids = {s: f"q{i}" for i, s in enumerate(pd.unique(pairs["sequence"]))}
        hits = diamond_hits({v: s for s, v in seq_ids.items()}, sequences_file, min_identity=min_identity,
                            coverage=coverage, threads=threads, diamond=diamond)
        hits = hits.sort_values(["query", "identity", "target"], ascending=[True, False, True])
        by_query = {q: list(zip(g["target"], g["identity"])) for q, g in hits.groupby("query", sort=False)}
        m = measurements.dropna(subset=["seq_id", "substrate_inchikey"])
        medians = m.groupby(["seq_id", m["substrate_inchikey"].str[:25]])["log10_kcat"].median().to_dict()

        estimates: dict[tuple[str, str], tuple[float, float, int]] = {}
        for seq, ik in pairs[["sequence", "inchikey"]].drop_duplicates().itertuples(index=False):
            found = [(ident, medians[(t, ik)]) for t, ident in by_query.get(seq_ids[seq], []) if (t, ik) in medians][:k]
            if found:
                w = np.array([f[0] for f in found])
                v = np.array([f[1] for f in found])
                estimates[(seq, ik)] = (float(np.sum(w * v) / np.sum(w)), float(w[0]), len(found))
        rows = []
        for (rxn, sub), g in pairs.groupby(["rxn_id", "substrate"], sort=False):
            est = [estimates[key] for key in dict.fromkeys(zip(g["sequence"], g["inchikey"])) if key in estimates]
            if not est:
                continue
            best = max(est, key=lambda e: e[1])
            rows.append({"rxn_id": rxn, "source": "homologs", "substrates": [sub],
                         "kcat": float(10 ** np.median([e[0] for e in est])),
                         "detail": f"identity={best[1]:.2f}; n={best[2]}"})
        kcat_list = pd.DataFrame(rows, columns=columns)
    if save:
        parameters = {"k": k, "min_identity": min_identity, "coverage": coverage}
        path = save_kcat_source(model, kcat_list, save_as, adapter=adapter, generator="kcats_from_homologs",
                                parameters=parameters, data=measurements.attrs.get("release", ""))
        logger.info("Homolog kcats for %d reactions written to %s", kcat_list["rxn_id"].nunique(), path)
    return kcat_list
