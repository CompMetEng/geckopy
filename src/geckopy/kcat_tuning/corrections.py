"""Report a tuning result as the corrections it makes.

A tuned kcat vector is one number per enzyme-reaction pair, and on a
genome-scale model that is thousands of numbers of which a few dozen
are supported by the data. Read as a vector it cannot be checked. Read
as a ranked list of changes, each carrying where its old value came
from and how much of the model's measured sensitivity it accounts for,
it can: a reviewer can look up whether one enzyme is really that slow.

The ranking is by *leverage* -- how far the distance moves when that
kcat alone is perturbed -- rather than by how far the tuning moved it.
Sorting by movement puts the parameters the data cannot see at the top,
since those are free to drift furthest. ``cumulative_share`` says how
much of the total leverage the list has accounted for by each row, so a
reader can stop where it stops climbing.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Sequence

import numpy as np

# Isozyme copies of one reaction are stored as separate ec rows with a
# suffix; the underlying reaction has the name.
_ISOZYME_SUFFIX = "_EXP_"


@dataclass(frozen=True)
class Correction:
    """One kcat the tuning changed, with its provenance."""

    rxn_id: str
    name: str
    ec_code: str
    source: str
    kcat_prior: float
    kcat_tuned: float
    fold_change: float
    leverage: float
    leverage_share: float
    cumulative_share: float


def _fold(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    return np.exp(np.abs(np.log(np.asarray(new, dtype=float) / old)))


def corrections(
    kcat: np.ndarray,
    kcat0: np.ndarray,
    rxn_ids: Sequence[str],
    *,
    sources: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
    ec_codes: Optional[Sequence[str]] = None,
    leverage: Optional[np.ndarray] = None,
    rel_tol: float = 0.02,
) -> list[Correction]:
    """The changed kcats, most consequential first.

    Parameters
    ----------
    kcat, kcat0
        Tuned and prior kcat vectors, same length as ``rxn_ids``.
    sources, names, ec_codes
        Per-kcat provenance. Missing entries are reported as ``""``
        rather than dropped, so a row never silently disappears.
    leverage
        Per-kcat effect on the distance from a one-at-a-time screen (see
        :func:`~.tuning.screen_kcat_leverage`). If given, rows are
        ordered by it and the share columns are filled; otherwise rows
        are ordered by fold change and the shares are 0.
    rel_tol
        A kcat within ``1 + rel_tol`` fold of its prior is unchanged.

    Returns
    -------
    list of :class:`Correction`, ordered most consequential first.
    """
    kcat = np.asarray(kcat, dtype=float)
    kcat0 = np.asarray(kcat0, dtype=float)
    if kcat.shape != kcat0.shape:
        raise ValueError(
            f"kcat has shape {kcat.shape}; kcat0 has {kcat0.shape}."
        )
    if len(rxn_ids) != len(kcat):
        raise ValueError(
            f"rxn_ids has {len(rxn_ids)} entries; kcat has {len(kcat)}."
        )

    fold = _fold(kcat, kcat0)
    changed = np.flatnonzero(fold > 1.0 + rel_tol)

    if leverage is None:
        lev = np.zeros(len(kcat))
        order = changed[np.argsort(fold[changed])[::-1]]
        total = 0.0
    else:
        lev = np.asarray(leverage, dtype=float)
        if lev.shape != kcat.shape:
            raise ValueError(
                f"leverage has shape {lev.shape}; expected {kcat.shape}."
            )
        order = changed[np.argsort(lev[changed])[::-1]]
        total = float(lev.sum())

    def at(seq, i):
        return "" if seq is None else str(seq[i])

    rows: list[Correction] = []
    running = 0.0
    for i in order:
        share = lev[i] / total if total > 0 else 0.0
        running += share
        rows.append(Correction(
            rxn_id=str(rxn_ids[i]),
            name=at(names, i),
            ec_code=at(ec_codes, i),
            source=at(sources, i),
            kcat_prior=float(kcat0[i]),
            kcat_tuned=float(kcat[i]),
            fold_change=float(fold[i]),
            leverage=float(lev[i]),
            leverage_share=float(share),
            cumulative_share=float(running),
        ))
    return rows


def corrections_tsv(rows: Sequence[Correction]) -> str:
    """The corrections as a TSV document, header included."""
    cols = [f.name for f in fields(Correction)]
    out = ["\t".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = getattr(r, c)
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        out.append("\t".join(vals))
    return "\n".join(out) + "\n"


def annotate_from_model(model, rxn_ids: Sequence[str]) -> dict[str, list[str]]:
    """Names, EC codes and sources for ``rxn_ids``, read off an ecModel.

    An isozyme copy has a ``_EXP_n`` suffix that the underlying
    reaction does not, so the name is looked up under the stripped id.
    Anything absent is reported as ``""``.
    """
    ec = model.ec
    by_id = {r: i for i, r in enumerate(ec.rxns)}
    names, ec_codes, sources = [], [], []
    for rxn_id in rxn_ids:
        i = by_id.get(rxn_id)
        ec_codes.append("" if i is None or ec.eccodes is None else str(ec.eccodes[i]))
        sources.append("" if i is None or ec.source is None else str(ec.source[i]))
        base = rxn_id.split(_ISOZYME_SUFFIX)[0]
        try:
            names.append(model.reactions.get_by_id(base).name or "")
        except (KeyError, AttributeError):
            names.append("")
    return {"names": names, "ec_codes": ec_codes, "sources": sources}
