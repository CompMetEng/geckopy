"""Measure how many free parameters a tuning run actually needs.

:func:`~.tuning.select_tunable_mask`'s ``target_impact_share`` selects a
comparable *share* of leverage on any model -- that is the point of a
relative cutoff over a fixed count or a fixed leverage value. It does not
follow that the same share sits at a comparable point on the *fit*
curve: how concentrated a model's leverage is, and how quickly its fit
stops improving as more parameters join the search, are different
properties of the model and the data, and they do not move together.
Two ecModels screened and tuned the same way can have leverage curves
of similar shape yet fit curves that diverge sharply -- one still
climbing at ``target_impact_share=0.9``, the other flat since a third of
the way there. The only way to tell them apart is to tune at a few
candidate sizes and look at where the improvement stops, which is what
this module automates.

The two functions here are meant to be used together but stay separate:
:func:`sweep_tunable_mask_size` only measures (it runs
:func:`~.tuning.cmaes_kcat_tuning` at each requested share and reports
what came back); :func:`recommend_target_impact_share` only decides,
from measurements already in hand. Passing a sweep's own points through
the second lets a different stopping rule be tried without re-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .tuning import (
    _resolve_context, cmaes_kcat_tuning, screen_kcat_leverage, select_tunable_mask,
)

if TYPE_CHECKING:
    from ...adapter import ModelAdapter
    from ...adapter.params import EvotuneParams
    from ...ec_model.ec_model import EcModel
    from .data import EvotuneData


@dataclass(frozen=True)
class MaskSizePoint:
    """Tuned fit at one candidate ``target_impact_share``.

    Attributes
    ----------
    target_impact_share
        The share this point was tuned at.
    n_selected
        Free parameters :func:`~.tuning.select_tunable_mask` chose at
        this share (tied isozyme copies count once).
    rmse
        The lowest value reached in each seed's
        ``EvotuneResult.rmse_trace`` -- not ``rmse_trace[-1]``, because
        ``rmse_trace`` only records plain RMSE at a step that improved
        the (possibly prior-penalised) *objective*, per
        ``cmaes_kcat_tuning``'s own docstring; with the default nonzero
        ``prior_penalty_weight`` a later generation can improve the
        objective while its plain RMSE is slightly worse than an
        earlier generation's, making the final entry not necessarily
        the best one recorded. ``min(rmse_trace)`` is the best that
        *was* recorded either way, though a candidate that improved
        plain RMSE without improving the objective is invisible to
        both and to ``rmse_trace`` itself -- set
        ``params.prior_penalty_weight=0`` before a sweep meant to
        measure fit alone, if that residual gap matters. Kept per-seed
        rather than pre-averaged so a caller can look at the spread
        directly.
    seeds
        The seeds each entry of ``rmse`` came from.
    """

    target_impact_share: float
    n_selected: int
    rmse: tuple[float, ...]
    seeds: tuple[int, ...]

    @property
    def rmse_mean(self) -> float:
        return float(np.mean(self.rmse))

    @property
    def rmse_sd(self) -> float:
        """Sample standard deviation across seeds; ``nan`` with fewer
        than two -- a single seed carries no information about noise,
        and pretending otherwise (e.g. reporting 0) understates it."""
        return float(np.std(self.rmse, ddof=1)) if len(self.rmse) > 1 else float("nan")


def sweep_tunable_mask_size(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    evotune_data: Optional["EvotuneData"] = None,
    okp_method: Optional[str] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    screen: Optional[pd.DataFrame] = None,
    target_impact_shares: Sequence[float] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95),
    seeds: Sequence[int] = (0, 1),
    n_proc: Optional[int] = None,
    verbose: bool = True,
) -> list[MaskSizePoint]:
    """Tune at each of ``target_impact_shares`` and report the fit.

    The screen is computed once (or reused, via ``screen``) and shared
    across every share and seed; only :func:`~.tuning.cmaes_kcat_tuning`
    reruns per point. Each run starts from ``model``'s own current
    ``ec.kcat`` -- ``model`` is not mutated (a fresh copy is tuned each
    time), unlike :func:`~.tuning.cmaes_kcat_tuning` itself.

    Cost is one screen (comparable to a full tuning run) plus
    ``len(target_impact_shares) * len(seeds)`` tuning runs, each sized
    by however many parameters that share selects -- the later points
    in an increasing sequence of shares cost more, both because CMA-ES
    scales its population with dimension and because a wider search
    space usually takes longer to explore.

    The defaults are a coarse scout, not a guarantee of finding the
    true optimum: five or six points cannot resolve a curve whose real
    knee sits inside one of the gaps between them, and how wide that
    gap needs to be to matter is exactly what a first sweep is for
    finding out. Look at ``n_selected`` after a first sweep -- if
    consecutive shares jump by a large multiple in parameter count
    (``target_impact_share`` is closer to the leverage tail than to its
    bulk there, e.g. the jump from 0.9 to 0.95 was 5x on one real
    model), or if the recommendation lands next to the widest such gap,
    that gap is where the true knee could be hiding; rerun with a few
    shares added inside it. The same applies at the low end: if the
    smallest share measured isn't clearly worse than the best one, it
    has not necessarily bracketed the true minimum, and a smaller share
    is worth adding to confirm it.

    ``seeds`` defaults to two, not one, doubling the cost of the
    default call: :func:`recommend_target_impact_share`'s noise-floor
    check needs at least two seeds at a point to have anything to
    check against, and with only one it silently falls back to a
    cruder rule everywhere. Pass a single seed for the very first,
    cheapest possible scout of the shape; widen to more once a
    promising region is known, since that is where seed noise is most
    likely to matter (fit values close enough together for it to
    decide the comparison).

    Parameters
    ----------
    model
        EcModel with a populated ``ec.kcat``, used as the tuning
        starting point at every share/seed. Not mutated.
    target_impact_shares
        Candidate shares to measure, any order (sorted on return).
    seeds
        Seeds tuned at every share.
    screen
        A precomputed :func:`~.tuning.screen_kcat_leverage` report.
        Computed once, fresh, if not given.
    adapter, params, evotune_data, okp_method, bio_rxn, make_anaerobic,
    change_protein_biomass, n_proc, verbose
        Forwarded to :func:`~.tuning.screen_kcat_leverage` and
        :func:`~.tuning.cmaes_kcat_tuning`; see their docstrings.

    Returns
    -------
    list of :class:`MaskSizePoint`, one per share, sorted by
    ``target_impact_share`` ascending.
    """
    params, evotune_data, bio_rxn, okp_method = _resolve_context(
        model, adapter, params, evotune_data, bio_rxn, okp_method)

    if screen is None:
        screen = screen_kcat_leverage(
            model, adapter=adapter, params=params, evotune_data=evotune_data,
            okp_method=okp_method, bio_rxn=bio_rxn,
            make_anaerobic=make_anaerobic, change_protein_biomass=change_protein_biomass,
            n_proc=n_proc,
        )

    points = []
    for share in sorted(target_impact_shares):
        mask = select_tunable_mask(model, screen, target_impact_share=share)
        n_selected = int(mask.sum())
        rmses = []
        for seed in seeds:
            run_model = model.copy()
            result = cmaes_kcat_tuning(
                run_model, adapter=adapter, params=params, evotune_data=evotune_data,
                okp_method=okp_method, bio_rxn=bio_rxn,
                make_anaerobic=make_anaerobic, change_protein_biomass=change_protein_biomass,
                tunable_mask=mask, n_proc=n_proc, seed=seed, verbose=False,
            )
            rmses.append(float(min(result.rmse_trace)))
            if verbose:
                print(
                    f"target_impact_share={share}: {n_selected} selected, "
                    f"seed {seed}: rmse {rmses[-1]:.4f}",
                    flush=True,
                )
        points.append(MaskSizePoint(
            target_impact_share=float(share), n_selected=n_selected,
            rmse=tuple(rmses), seeds=tuple(seeds),
        ))
    return points


def _local_sd(a: MaskSizePoint, b: MaskSizePoint) -> Optional[float]:
    """Pool ``a`` and ``b``'s own seed sd, using whichever of the two
    actually has one (both, if both do). Deliberately not a sweep-wide
    pooled sd: how noisy a comparison is can genuinely differ across
    the sweep (e.g. a larger parameter count giving CMA-ES more room to
    land on different local optima across seeds), and borrowing an
    unrelated point's spread to judge this pair would let noise from
    somewhere else in the sweep, that has nothing to do with this
    specific comparison, decide it."""
    sds = [p.rmse_sd for p in (a, b) if not np.isnan(p.rmse_sd)]
    if not sds:
        return None
    return float(np.sqrt(np.mean(np.square(sds))))


def recommend_target_impact_share(
    points: Sequence[MaskSizePoint],
    *,
    rel_tol: float = 0.02,
    noise_multiplier: float = 2.0,
) -> tuple[float, str]:
    """Pick the smallest share past which tuning stops paying for itself.

    Walks ``points`` in increasing share order and returns the first
    one, ``p``, whose best later point does not clearly improve on it.
    "Clearly" is two independent tests, either enough on its own to
    call a gain not worth it: the gain is small in relative terms
    (within ``rel_tol`` of ``p``'s own RMSE), or -- only checkable where
    at least one of the two points being compared has more than one
    seed -- the gain is small next to how much repeating just those two
    points varies (within ``noise_multiplier`` times their own pooled
    seed standard deviation, not a sweep-wide one; see :func:`_local_sd`).
    A comparison where neither point has a repeated seed only gets the
    relative check, which alone cannot tell a genuine plateau from a
    lucky seed -- ``reason`` says which test(s) applied so that
    limitation is visible rather than silent.

    Parameters
    ----------
    points
        Output of :func:`sweep_tunable_mask_size` (or hand-built), at
        least two entries, any order (sorted internally).
    rel_tol
        A later point counts as "no better" when its RMSE is within
        this fraction of the earlier one's.
    noise_multiplier
        How many local standard deviations a gain must clear, where
        checkable, to count as more than noise.

    Returns
    -------
    (share, reason)
        The recommended ``target_impact_share`` and one sentence on
        why -- always the ``target_impact_share`` of one of ``points``,
        since only shares actually measured can be recommended.

    Raises
    ------
    ValueError
        If ``points`` has fewer than two entries.
    """
    pts = sorted(points, key=lambda p: p.target_impact_share)
    if len(pts) < 2:
        raise ValueError(
            f"Need at least two points to compare, got {len(pts)}. "
            "sweep_tunable_mask_size needs at least two target_impact_shares."
        )

    for i, p in enumerate(pts[:-1]):
        best_later = min(pts[i + 1:], key=lambda q: q.rmse_mean)
        gain = p.rmse_mean - best_later.rmse_mean
        rel_gain = gain / p.rmse_mean if p.rmse_mean > 0 else 0.0
        rel_ok = rel_gain <= rel_tol
        sd = _local_sd(p, best_later)
        noise_ok = sd is not None and gain <= noise_multiplier * sd

        if rel_ok or noise_ok:
            reasons = []
            if rel_ok:
                reasons.append(
                    f"only {100 * rel_gain:.1f}% relative "
                    f"(<= {100 * rel_tol:.0f}% tolerance)"
                )
            if noise_ok:
                reasons.append(
                    f"{gain:.4g} absolute, within {noise_multiplier:g}x the "
                    f"local seed sd ({sd:.4g}) -- not distinguishable from "
                    "seed noise"
                )
            elif sd is None:
                reasons.append(
                    "no repeated seed at either point to check against noise"
                )
            return p.target_impact_share, (
                f"share {best_later.target_impact_share:g}'s best improvement "
                f"over this point's {p.rmse_mean:.4g} is " + " and ".join(reasons)
            )

    return pts[-1].target_impact_share, (
        "RMSE kept improving beyond every share measured; "
        "widen target_impact_shares to find where it actually flattens"
    )
