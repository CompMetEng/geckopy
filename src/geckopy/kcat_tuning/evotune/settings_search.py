"""Try several tuning settings and suggest a combination.

Which kcats to search (the mask), how strongly to charge for moving a
kcat away from its prior, and how many generations to allow all change
what :func:`~.tuning.cmaes_kcat_tuning` returns, and the best values
depend on the model and the data. :func:`search_tuning_settings` tunes at
every candidate and seed, :func:`summarise_runs` reduces the runs to one
row per candidate, and :func:`recommend_tuning_settings` picks from that
table.

A candidate is a dict with any of these keys:

``target_impact_share``, ``min_leverage``
    The tunable set, as in :func:`~.tuning.select_tunable_mask`.
    ``min_leverage`` takes precedence when both are given.
``max_generations`` and any other :class:`~geckopy.adapter.params.EvotuneParams`
    field except those listed next
    The search settings.

``max_growth_weight`` cannot be varied. It changes how the plain RMSE is
defined, so the fits of candidates that differ in it cannot be compared;
choose it from what the conditions are meant to weigh and keep it fixed.

Nothing here withholds conditions. A candidate is judged on the criteria
that need no spare data: the fit of the tuning conditions, how many kcats
the run changed, and whether repeating the run with another seed moves
the same kcats in the same direction. Fit alone always favours the
largest set of free kcats, because each one added can only lower the
RMSE of the conditions it is tuned on. A recommendation that rests on fit
alone therefore says only how many kcats the data can use, not how many
of them are determined by it; the reproducibility column is what limits
it. Use two or more seeds; with one, the reproducibility criterion is
skipped and the recommendation says so.

The cost is ``len(candidates) * len(seeds)`` tuning runs. Each run is
independent, so a cluster can run them as separate jobs through
:func:`run_tuning_settings` and combine them with
:func:`summarise_runs`.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .parsimony import fold_change, n_changed
from .tuning import (
    _resolve_context, cmaes_kcat_tuning, screen_kcat_leverage, select_tunable_mask,
)

if TYPE_CHECKING:
    from ...adapter import ModelAdapter
    from ...adapter.params import EvotuneParams
    from ...ec_model.ec_model import EcModel
    from ...experiments.data import ExperimentData

logger = logging.getLogger(__name__)

#: Keys of a candidate that choose the tunable set.
MASK_KEYS = ("target_impact_share", "min_leverage")
#: ``EvotuneParams`` fields that change the definition of the RMSE.
_METRIC_FIELDS = ("max_growth_weight",)


@dataclass
class SettingsRun:
    """One tuning run at one candidate and seed."""

    settings: dict
    seed: int
    n_selected: int
    old_kcat: np.ndarray
    new_kcat: np.ndarray
    rmse_trace: list[float]
    n_generations: int
    converged: bool
    max_generations: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def rmse(self) -> float:
        return float(self.rmse_trace[-1]) if self.rmse_trace else float("nan")


def candidate_grid(**values: Sequence) -> list[dict]:
    """Every combination of the given values, as a list of candidates.

    ``candidate_grid(min_leverage=[0.02, 0.03], prior_penalty_weight=[0, 0.03])``
    returns four candidates.
    """
    keys = list(values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*values.values())]


def _check_candidates(candidates: Sequence[Mapping], params: "EvotuneParams") -> None:
    allowed = set(MASK_KEYS) | set(type(params).model_fields)
    for cand in candidates:
        for key in cand:
            if key in _METRIC_FIELDS:
                raise ValueError(
                    f"{key!r} changes how the RMSE is defined, so candidates that "
                    "differ in it cannot be compared. Keep it fixed in params."
                )
            if key not in allowed:
                raise ValueError(
                    f"Unknown setting {key!r}. Use {', '.join(MASK_KEYS)} or a "
                    "field of EvotuneParams."
                )


def run_tuning_settings(
    model: "EcModel",
    settings: Mapping,
    seed: int,
    *,
    screen: pd.DataFrame,
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    experiment_data: Optional["ExperimentData"] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    popsize: Optional[int] = None,
    n_proc: Optional[int] = None,
) -> SettingsRun:
    """Tune once at one candidate and one seed.

    ``model`` is not changed: a copy is tuned. ``screen`` is a
    :func:`~.tuning.screen_kcat_leverage` report, computed once by the
    caller and shared by every run.
    """
    params, experiment_data, bio_rxn = _resolve_context(
        model, adapter, params, experiment_data, bio_rxn)
    _check_candidates([settings], params)
    settings = dict(settings)
    run_params = params.model_copy(update={
        k: v for k, v in settings.items() if k not in MASK_KEYS})
    mask = select_tunable_mask(
        model, screen,
        target_impact_share=settings.get("target_impact_share", 0.9),
        min_leverage=settings.get("min_leverage"),
        sigma_ref=run_params.sigma0_log_default,
    )
    start = time.monotonic()
    result = cmaes_kcat_tuning(
        model.copy(), adapter=adapter, params=run_params, experiment_data=experiment_data,
        bio_rxn=bio_rxn, make_anaerobic=make_anaerobic,
        change_protein_biomass=change_protein_biomass, tunable_mask=mask,
        popsize=popsize, n_proc=n_proc, seed=seed, verbose=False,
    )
    return SettingsRun(
        settings=settings, seed=int(seed), n_selected=int(mask.sum()),
        old_kcat=np.asarray(result.old_kcat), new_kcat=np.asarray(result.new_kcat),
        rmse_trace=list(result.rmse_trace), n_generations=int(result.n_generations),
        converged=bool(result.converged), max_generations=int(run_params.max_generations),
        seconds=time.monotonic() - start,
    )


def _plateau_generation(trace: Sequence[float], rel_tol: float) -> int:
    """First generation (1-based) whose best RMSE is within ``rel_tol`` of the final one."""
    if not len(trace):
        return 0
    final = trace[-1]
    for i, value in enumerate(trace):
        if value <= final * (1.0 + rel_tol):
            return i + 1
    return len(trace)


def _agreement(runs: Sequence[SettingsRun], fold_threshold: float) -> tuple[int, float]:
    """Kcats moved past ``fold_threshold`` by two seeds, and the share that
    went the same way, pooled over every pair of seeds."""
    n_both = n_agree = 0
    for a, b in itertools.combinations(runs, 2):
        if a.old_kcat.shape != b.old_kcat.shape:
            continue
        fa = fold_change(a.new_kcat, a.old_kcat)
        fb = fold_change(b.new_kcat, b.old_kcat)
        both = (fa > fold_threshold) & (fb > fold_threshold)
        same = np.sign(np.log(a.new_kcat[both] / a.old_kcat[both])) == \
            np.sign(np.log(b.new_kcat[both] / b.old_kcat[both]))
        n_both += int(both.sum())
        n_agree += int(same.sum())
    return n_both, (n_agree / n_both if n_both else float("nan"))


def summarise_runs(
    runs: Sequence[SettingsRun],
    *,
    fold_threshold: float = 2.0,
    plateau_tol: float = 0.02,
) -> pd.DataFrame:
    """One row per candidate: fit, size, and agreement between seeds.

    Columns: the candidate's settings, ``n_selected`` (free kcats before
    tying), ``n_seeds``, ``rmse_mean`` and ``rmse_sd`` (``nan`` with one
    seed), ``n_changed_mean``, ``n_both_moved`` (kcats past
    ``fold_threshold`` in two seeds) and ``pct_direction_agree`` of them
    (``nan`` without a pair), ``plateau_generation`` (the latest, over
    seeds, at which the RMSE is within ``plateau_tol`` of its final
    value), ``capped`` (any seed used every allowed generation, so the
    plateau is a lower bound) and ``seconds`` (mean per run).
    """
    groups: dict[str, list[SettingsRun]] = {}
    for run in runs:
        groups.setdefault(repr(sorted(run.settings.items())), []).append(run)
    rows = []
    for group in groups.values():
        rmse = np.array([r.rmse for r in group])
        n_both, agree = _agreement(group, fold_threshold)
        row = dict(group[0].settings)
        row.update({
            "n_selected": group[0].n_selected,
            "n_seeds": len(group),
            "rmse_mean": float(rmse.mean()),
            "rmse_sd": float(rmse.std(ddof=1)) if len(group) > 1 else float("nan"),
            "n_changed_mean": float(np.mean(
                [n_changed(r.new_kcat, r.old_kcat) for r in group])),
            "n_both_moved": n_both,
            "pct_direction_agree": agree,
            "plateau_generation": max(
                _plateau_generation(r.rmse_trace, plateau_tol) for r in group),
            "capped": any(
                not r.converged and r.n_generations >= r.max_generations
                for r in group),
            "seconds": float(np.mean([r.seconds for r in group])),
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(["n_selected", "rmse_mean"]).reset_index(drop=True)


def search_tuning_settings(
    model: "EcModel",
    candidates: Sequence[Mapping],
    *,
    seeds: Sequence[int] = (0, 1),
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    experiment_data: Optional["ExperimentData"] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    screen: Optional[pd.DataFrame] = None,
    popsize: Optional[int] = None,
    n_proc: Optional[int] = None,
    verbose: bool = True,
    **summary_kwargs,
) -> tuple[pd.DataFrame, list[SettingsRun]]:
    """Tune at every candidate and seed; return the summary and the runs.

    The screen is computed once, or taken from ``screen``. ``model`` is
    not changed. See the module docstring for the candidates and for
    what the summary can and cannot say; pass the table to
    :func:`recommend_tuning_settings`.
    """
    params, experiment_data, bio_rxn = _resolve_context(
        model, adapter, params, experiment_data, bio_rxn)
    _check_candidates(candidates, params)
    if screen is None:
        screen = screen_kcat_leverage(
            model, adapter=adapter, params=params, experiment_data=experiment_data,
            bio_rxn=bio_rxn, make_anaerobic=make_anaerobic,
            change_protein_biomass=change_protein_biomass, n_proc=n_proc,
        )
    runs = []
    for cand in candidates:
        for seed in seeds:
            run = run_tuning_settings(
                model, cand, seed, screen=screen, adapter=adapter, params=params,
                experiment_data=experiment_data, bio_rxn=bio_rxn,
                make_anaerobic=make_anaerobic,
                change_protein_biomass=change_protein_biomass,
                popsize=popsize, n_proc=n_proc,
            )
            runs.append(run)
            if verbose:
                logger.info("%s seed %d: %d selected, rmse %.4f, %d generations",
                            dict(cand), seed, run.n_selected, run.rmse,
                            run.n_generations)
    return summarise_runs(runs, **summary_kwargs), runs


def recommend_tuning_settings(
    table: pd.DataFrame,
    *,
    rel_tol: float = 0.02,
    noise_multiplier: float = 2.0,
    min_agreement: float = 0.8,
    min_both_moved: int = 5,
) -> tuple[dict, str]:
    """Pick the smallest candidate whose fit is close to the best that is reproducible.

    1. A candidate whose seeds disagree is dropped: at least
       ``min_both_moved`` kcats moved past the fold threshold in two
       seeds, and fewer than ``min_agreement`` of them in the same
       direction. A candidate with fewer such kcats, or with a single
       seed, passes, and the reason says that the check was not made.
    2. Among the rest, the best fit is the lowest ``rmse_mean``. A
       candidate is close to it when its ``rmse_mean`` is within
       ``rel_tol`` of the best, or within ``noise_multiplier`` times the
       pooled seed sd of the two.
    3. Of the close candidates, the one with the fewest changed kcats
       wins, then the fewest selected, then the lowest RMSE.

    Returns the candidate (its setting columns only) and one sentence on
    why. ``max_generations`` is added to it from ``plateau_generation``
    (25% above the plateau) when the candidate did not set it and no
    seed was capped.
    """
    if table.empty:
        raise ValueError("table has no rows")
    setting_cols = [c for c in table.columns if c in MASK_KEYS or c not in _RESULT_COLUMNS]
    notes = []
    ok = table.copy()
    checked = (ok["n_both_moved"] >= min_both_moved) & ok["pct_direction_agree"].notna()
    bad = checked & (ok["pct_direction_agree"] < min_agreement)
    if bad.any():
        notes.append(f"{int(bad.sum())} candidate(s) dropped for seed disagreement "
                     f"(under {100 * min_agreement:.0f}% same direction)")
    ok = ok[~bad]
    if ok.empty:
        raise ValueError(
            "Every candidate failed the reproducibility check; the seeds disagree "
            "on the kcats they move. Raise prior_penalty_weight or shrink the mask."
        )
    if not checked[ok.index].any():
        notes.append("reproducibility could not be checked (one seed, or too few "
                     "kcats moved past the threshold)")
    best = ok.loc[ok["rmse_mean"].idxmin()]
    close = []
    for idx, row in ok.iterrows():
        gain = row["rmse_mean"] - best["rmse_mean"]
        sds = [v for v in (row["rmse_sd"], best["rmse_sd"]) if not np.isnan(v)]
        sd = float(np.sqrt(np.mean(np.square(sds)))) if sds else None
        if gain <= rel_tol * best["rmse_mean"] or (
                sd is not None and gain <= noise_multiplier * sd):
            close.append(idx)
    chosen = ok.loc[close].sort_values(
        ["n_changed_mean", "n_selected", "rmse_mean"]).iloc[0]
    cand = {c: chosen[c] for c in setting_cols if not pd.isna(chosen[c])}
    cand = {k: (v.item() if hasattr(v, "item") else v) for k, v in cand.items()}
    if "max_generations" not in cand and not chosen["capped"]:
        cand["max_generations"] = int(np.ceil(1.25 * chosen["plateau_generation"]))
    reason = (
        f"fewest changed kcats ({chosen['n_changed_mean']:.0f}) among the "
        f"{len(close)} candidate(s) within {100 * rel_tol:.0f}% or "
        f"{noise_multiplier:g} seed sd of the best RMSE ({best['rmse_mean']:.4g}); "
        f"its own is {chosen['rmse_mean']:.4g}"
    )
    if notes:
        reason += "; " + "; ".join(notes)
    return cand, reason


_RESULT_COLUMNS = frozenset({
    "n_selected", "n_seeds", "rmse_mean", "rmse_sd", "n_changed_mean",
    "n_both_moved", "pct_direction_agree", "plateau_generation", "capped",
    "seconds",
})
