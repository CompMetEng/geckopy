"""Flag kcat assignments that deserve a second look.

The tuner's largest corrections are rarely discoveries about biology.
They are places the assignment went wrong: a value derived from one
specific-activity measurement, a database maximum standing in for a
distribution with a two-hundred-fold spread, one fallback number reused
across a whole family of reactions. Finding those directly is cheaper
than inferring them from a fit, and a corrected prior propagates to
every model built from the same databases, where a tuned kcat stays in
the model it was tuned on.

Two design choices, both from measurement rather than taste.

**Rank by leverage, and cut the report there.** On ecYeastGEM the
checks below flag around 1900 of 4834 kcats, which is a report nobody
reads. Tightening the checks does not fix that: raising the EC-maximum
gap from three-fold to ten-fold still returns 216 entries and drops
*all six* of the high-leverage ones, because looking odd and mattering
are unrelated properties. Filtering by leverage keeps the list short
and keeps the entries worth a curator's time.

**Report a parameter once.** Isozyme copies the assignment could not
separate share a value and a source, so they share a finding; listing
them separately fills the report with duplicates of one decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ..kcat_tuning.evotune.tying import (
    base_reaction, isozyme_tie_map,
)

#: Checks reported unless asked for explicitly. ``standard`` values are a
#: deliberate fallback and ``tied`` groups are unresolvable by design, so
#: listing every one of them is noise rather than news.
OPTIONAL_CHECKS = ("standard-fallback", "tied-isozymes")


@dataclass(frozen=True)
class Finding:
    """One kcat assignment worth a closer look."""

    rxn_id: str
    name: str
    ec_code: str
    source: str
    kcat: float
    leverage: float
    n_isozymes: int
    other_reactions: int
    checks: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class EcStats:
    """What the database holds for one EC number."""

    n_kcat: int = 0
    kcat_max: float = float("nan")
    kcat_median: float = float("nan")
    values: frozenset = frozenset()
    n_sa: int = 0


def _first_ec(ec_code: str) -> str:
    return str(ec_code).split(";")[0].strip()


def review_assignment(
    rxn_ids: Sequence[str],
    kcats: np.ndarray,
    sources: Sequence[str],
    ec_codes: Sequence[str],
    *,
    ec_stats: Optional[Mapping[str, EcStats]] = None,
    enzymes: Optional[Sequence[Iterable[str]]] = None,
    names: Optional[Sequence[str]] = None,
    leverage: Optional[np.ndarray] = None,
    slow: float = 1e-2,
    fast: float = 1e4,
    ec_max_gap: float = 3.0,
    repeat_min: int = 5,
    include: Sequence[str] = (),
    coverage: Optional[float] = None,
    top: Optional[int] = None,
) -> list[Finding]:
    """Assignments worth reviewing, most consequential first.

    Parameters
    ----------
    ec_stats
        Per-EC database summary. Checks needing it are skipped when it
        is absent, so the function still runs without a database.
    enzymes
        Protein ids per kcat, used to count how many other reactions
        share an enzyme -- a change here may imply changes there.
    leverage
        Per-kcat effect on the distance from a one-at-a-time screen.
        Without it the findings cannot be ranked and are returned in
        input order.
    coverage, top
        Truncate the report. ``coverage`` keeps the rows carrying that
        share of the *flagged* leverage, so 0.9 drops the tail that
        together accounts for a tenth of what the findings are worth;
        it adapts to the model where ``top`` does not. Measured against
        all kcats instead, the threshold would be unreachable, since
        flagged rows are a minority of the model's leverage.
    include
        Names from :data:`OPTIONAL_CHECKS` to report as well.
    sources
        The source labels ``custom`` and ``standard`` are recognised
        without regard to letter case.

    Returns
    -------
    list of :class:`Finding`, one per free parameter.
    """
    kcats = np.asarray(kcats, dtype=float)
    n = len(kcats)
    for name, seq in (("sources", sources), ("ec_codes", ec_codes),
                      ("rxn_ids", rxn_ids)):
        if len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} entries; kcats has {n}.")
    lev = np.zeros(n) if leverage is None else np.asarray(leverage, dtype=float)
    if lev.shape != kcats.shape:
        raise ValueError(f"leverage has shape {lev.shape}; expected {kcats.shape}.")

    tie = isozyme_tie_map(rxn_ids, kcats, sources)
    members = defaultdict(list)
    for i, rep in enumerate(tie):
        members[int(rep)].append(i)

    # A value reused across many distinct reactions is a fallback wearing
    # a measurement's label, whatever its source says.
    per_reaction = defaultdict(set)
    for i, rxn_id in enumerate(rxn_ids):
        per_reaction[round(float(np.log(kcats[i])), 9) if kcats[i] > 0 else None
                     ].add(base_reaction(str(rxn_id)))
    reuse = {k: len(v) for k, v in per_reaction.items()}

    enzyme_rxns = defaultdict(set)
    if enzymes is not None:
        for i, prots in enumerate(enzymes):
            for p in prots:
                enzyme_rxns[p].add(base_reaction(str(rxn_ids[i])))

    out: list[Finding] = []
    for rep, idx in members.items():
        i = idx[0]
        ec = _first_ec(ec_codes[i])
        st = (ec_stats or {}).get(ec)
        checks, detail = [], []

        if st is not None and st.n_kcat == 0:
            checks.append("no-ec-evidence")
            detail.append(
                f"no kcat rows for EC {ec}"
                + (f"; {st.n_sa} specific-activity rows instead" if st.n_sa else "")
            )
        key = round(float(np.log(kcats[i])), 9) if kcats[i] > 0 else None
        if reuse.get(key, 0) >= repeat_min:
            checks.append("repeated-value")
            detail.append(f"{kcats[i]:g} 1/s reused across {reuse[key]} reactions")
        if (st is not None and st.n_kcat and np.isfinite(st.kcat_median)
                and st.kcat_median > 0
                and abs(kcats[i] - st.kcat_max) <= 1e-9 * max(kcats[i], 1.0)
                and st.kcat_max / st.kcat_median > ec_max_gap):
            checks.append("ec-maximum")
            detail.append(
                f"took the EC maximum, {st.kcat_max / st.kcat_median:.0f}x its "
                f"median over {st.n_kcat} rows"
            )
        if kcats[i] < slow or kcats[i] > fast:
            checks.append("magnitude")
            detail.append(f"{kcats[i]:g} 1/s is outside {slow:g}-{fast:g}")
        if (str(sources[i]).lower() == "custom" and st is not None
                and round(float(kcats[i]), 6) in st.values):
            checks.append("custom-duplicate")
            detail.append("custom value equals a database value for its own EC")
        if len(idx) > 1:
            if "tied-isozymes" in include:
                checks.append("tied-isozymes")
            detail.append(f"{len(idx)} isozyme copies share this value")
        if str(sources[i]).lower() == "standard":
            if "standard-fallback" not in include:
                continue
            checks.append("standard-fallback")
            detail.append("the model's fallback value, not a measurement")

        if not checks:
            continue
        others = set()
        if enzymes is not None:
            for p in enzymes[i]:
                others |= enzyme_rxns[p]
        others.discard(base_reaction(str(rxn_ids[i])))
        out.append(Finding(
            rxn_id=str(rxn_ids[i]),
            name="" if names is None else str(names[i]),
            ec_code=ec,
            source=str(sources[i]),
            kcat=float(kcats[i]),
            leverage=float(lev[idx].max()),
            n_isozymes=len(idx),
            other_reactions=len(others),
            checks=tuple(checks),
            detail="; ".join(detail),
        ))

    out.sort(key=lambda f: -f.leverage)
    flagged_total = sum(f.leverage for f in out)
    if coverage is not None and flagged_total > 0:
        total, kept, acc = flagged_total, [], 0.0
        for f in out:
            kept.append(f)
            acc += f.leverage
            if acc / total >= coverage:
                break
        out = kept
    if top is not None:
        out = out[:top]
    return out


def ec_stats_from_brenda(brenda) -> dict[str, EcStats]:
    """Summarise a :class:`BrendaData` per EC number."""
    stats: dict[str, EcStats] = {}
    by_ec = {ec: g for ec, g in brenda.kcat_max.groupby("ec_code")}
    med = {ec: g for ec, g in brenda.kcat_median.groupby("ec_code")}
    sa = Counter(brenda.sa_max["ec_code"]) if len(brenda.sa_max) else Counter()
    for ec in set(by_ec) | set(sa):
        g = by_ec.get(ec)
        vals = set()
        if g is not None:
            vals |= {round(float(v), 6) for v in g["kcat"]}
        if ec in med:
            vals |= {round(float(v), 6) for v in med[ec]["kcat"]}
        stats[ec] = EcStats(
            n_kcat=0 if g is None else len(g),
            kcat_max=float("nan") if g is None else float(g["kcat"].max()),
            kcat_median=float("nan") if ec not in med
            else float(med[ec]["kcat"].median()),
            values=frozenset(vals),
            n_sa=int(sa.get(ec, 0)),
        )
    return stats


def findings_tsv(rows: Sequence[Finding]) -> str:
    """The findings as a TSV document, header included."""
    cols = [f.name for f in fields(Finding)]
    out = ["\t".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = getattr(r, c)
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            elif isinstance(v, tuple):
                vals.append(",".join(v))
            else:
                vals.append(str(v))
        out.append("\t".join(vals))
    return "\n".join(out) + "\n"
