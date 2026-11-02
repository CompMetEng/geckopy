"""Trading fit against how many kcats were moved.

A tuned kcat vector that fits well by rewriting every parameter is not
a useful answer: changes should be few, and concentrated in the
parameters we are least confident about. Movement is measured here in
units of each kcat's own prior sigma -- MATLAB's ``devFromPrior`` --
which weights by confidence automatically, since ``sigma0_log`` is
smaller for trusted sources.

The tuner returns the best-RMSE particle, which on the full ecYeastGEM
model moves ~99% of kcats by ~8 prior sigmas. Reverting the least-moved
of those to their priors costs nothing up to about 3 sigma and beats
the shrunk "blend" estimate at every sparsity level, so a frontier is a
more honest report than either single vector. See
``docs/internal/matlab_replication_results.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FrontierPoint:
    """One (parsimony, fit) trade-off, from :func:`parsimony_frontier`."""

    threshold: float
    kcat: np.ndarray
    n_changed: int
    mean_dev: float
    rmse: float


def fold_change(kcat: np.ndarray, kcat0: np.ndarray) -> np.ndarray:
    """Per-kcat fold change from the prior, always >= 1.

    ``max(k/k0, k0/k)``. Reported in preference to
    :func:`movement_in_sigma` because ``sigma0_log`` differs by source
    (0.10 for ``custom`` up to 0.30 for unlabelled), so dividing by it
    hides the per-source behaviour: a vector that moves every source by
    the same number of sigmas has moved the trusted ones three times
    *less* in fold terms. Only the *ranking* of the sigmas across
    sources is meaningful, not their absolute values.
    """
    return np.exp(np.abs(np.log(np.asarray(kcat, dtype=float) / kcat0)))


def source_movement(
    kcat: np.ndarray,
    kcat0: np.ndarray,
    groups: np.ndarray,
    order: Sequence[str],
    near_tol: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Median fold change and near-prior fraction, per source group.

    Two medians are reported per source, separating how *many* kcats a
    result moves from how *far* it moves them. ``median_fold`` is taken
    over every kcat of the source; where most of a source sits at its
    prior -- which a restrictive ``tunable_mask`` guarantees -- it
    saturates at 1.0 and carries no information about the tier.
    ``median_fold_moved`` conditions on the kcats that did move, and so
    stays informative at any sparsity.

    Parameters
    ----------
    order
        Source names from most to least trusted. Used only to report
        whether movement respects that ranking.
    near_tol
        A kcat within ``1 + near_tol`` fold of its prior counts as
        untouched, and is excluded from ``median_fold_moved``.

    Returns
    -------
    dict keyed by source name, plus ``"_ordered"``: whether
    ``median_fold_moved`` is non-decreasing along ``order``, i.e.
    whether the result moved trusted kcats less far than untrusted
    ones. A source with nothing moved reports ``median_fold_moved``
    1.0 and takes no part in that check.
    """
    fold = fold_change(kcat, kcat0)
    groups = np.asarray(groups)
    out: dict[str, dict[str, float]] = {}
    medians = []
    for name in order:
        sel = groups == name
        if not sel.any():
            continue
        moved = sel & (fold >= 1.0 + near_tol)
        med_moved = float(np.median(fold[moved])) if moved.any() else 1.0
        if moved.any():
            medians.append(med_moved)
        out[name] = {
            "median_fold": float(np.median(fold[sel])),
            "median_fold_moved": med_moved,
            "frac_near_prior": float(np.mean(fold[sel] < 1.0 + near_tol)),
            "n": int(sel.sum()),
            "n_moved": int(moved.sum()),
        }
    out["_ordered"] = all(
        medians[i] <= medians[i + 1] for i in range(len(medians) - 1)
    )
    return out


def impact_share(
    kcat: np.ndarray,
    kcat0: np.ndarray,
    drmse: np.ndarray,
    rel_tol: float = 0.02,
) -> float:
    """Fraction of the total screened impact carried by changed kcats.

    ``drmse`` is each kcat's one-at-a-time effect on the distance. A
    result that changes many parameters the data barely responds to
    scores low here even when it changes few of them; one that changes
    a handful of load-bearing parameters scores high. Distinguishes
    "few, large, and consequential" from mere sparsity -- on the full
    ecYeastGEM run the shrunk blend changes 560 kcats but carries only
    28% of the available impact, while 301 changes chosen by the screen
    carry 69%.
    """
    drmse = np.asarray(drmse, dtype=float)
    total = drmse.sum()
    if total <= 0:
        return 0.0
    changed = fold_change(kcat, kcat0) > (1.0 + rel_tol)
    return float(drmse[changed].sum() / total)


def identifiability_mask(
    drmse: np.ndarray,
    sigma0_log: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Which kcats have earned the right to be tuned.

    Keeps a kcat when ``|dRMSE| * sigma0_log > threshold``, where
    ``dRMSE`` is its one-at-a-time effect on the distance. Multiplying
    by ``sigma0_log`` makes a trusted source clear a proportionally
    higher bar before it may move: with ``custom`` at 0.10 and
    unlabelled at 0.30, a ``custom`` kcat needs three times the
    measured effect of an unlabelled one to qualify.

    This uses only the *ranking* of the sigmas across sources, which is
    the part that carries meaning -- their absolute values are a
    convention. One global ``threshold`` sets the overall strictness;
    the per-source differentiation is derived, not tuned.
    """
    return (np.asarray(drmse, dtype=float)
            * np.asarray(sigma0_log, dtype=float)) > threshold


def movement_in_sigma(
    kcat: np.ndarray, kcat0: np.ndarray, sigma0_log: np.ndarray,
) -> np.ndarray:
    """Per-kcat displacement from the prior, in prior standard deviations.

    ``|log(kcat / kcat0)| / sigma0_log``, matching MATLAB's
    ``devFromPrior = abs(muLog - log(kcat0)) ./ sigma0log``.
    """
    return np.abs(np.log(np.asarray(kcat, dtype=float) / kcat0)) / sigma0_log


def revert_below(
    kcat: np.ndarray,
    kcat0: np.ndarray,
    sigma0_log: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Put every kcat that moved less than ``threshold`` sigma back on
    its prior, leaving the rest at their tuned values."""
    keep = movement_in_sigma(kcat, kcat0, sigma0_log) >= threshold
    return np.where(keep, np.asarray(kcat, dtype=float), kcat0)


def n_changed(
    kcat: np.ndarray, kcat0: np.ndarray, rel_tol: float = 0.02,
) -> int:
    """How many kcats differ from their prior by more than ``rel_tol``.

    The 2% default matches the threshold MATLAB reports its per-source
    "unchanged" percentages at.
    """
    return int(
        np.count_nonzero(
            np.abs(np.log(np.asarray(kcat, dtype=float) / kcat0))
            > np.log1p(rel_tol)
        )
    )


def parsimony_frontier(
    score: Callable[[np.ndarray], float],
    kcat: np.ndarray,
    kcat0: np.ndarray,
    sigma0_log: np.ndarray,
    thresholds: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0),
) -> list[FrontierPoint]:
    """Score ``kcat`` with progressively more of it reverted to prior.

    Parameters
    ----------
    score
        Scores one kcat vector; one call per threshold.
    kcat, kcat0, sigma0_log
        Tuned vector, prior vector, and per-kcat prior sigma.
    thresholds
        Sigma cutoffs to sweep, ascending. ``0.0`` reverts nothing and
        so reproduces ``kcat``'s own score.

    Returns
    -------
    list of FrontierPoint, in the order given.
    """
    kcat0 = np.asarray(kcat0, dtype=float)
    sigma0_log = np.asarray(sigma0_log, dtype=float)
    points = []
    for t in thresholds:
        vec = revert_below(kcat, kcat0, sigma0_log, t)
        points.append(FrontierPoint(
            threshold=float(t),
            kcat=vec,
            n_changed=n_changed(vec, kcat0),
            mean_dev=float(movement_in_sigma(vec, kcat0, sigma0_log).mean()),
            rmse=float(score(vec)),
        ))
    return points


def best_parsimonious(
    points: Sequence[FrontierPoint], tolerance: float = 0.02,
) -> FrontierPoint:
    """The fewest-changes point whose RMSE is within ``tolerance`` of the
    best on the frontier.

    Defaults to 2%, which is the scale of the alternate-optimum noise
    measured on this model -- below that, two vectors are not
    meaningfully different in fit, so the sparser one wins. A point with
    a NaN RMSE is never chosen and does not set the best RMSE; when every
    RMSE is NaN no point qualifies and ``ValueError`` is raised.
    """
    if not points:
        raise ValueError("points is empty.")
    rmses = [p.rmse for p in points if not np.isnan(p.rmse)]
    if not rmses:
        raise ValueError("every point has a NaN rmse.")
    best_rmse = min(rmses)
    within = [p for p in points if p.rmse <= best_rmse * (1.0 + tolerance)]
    return min(within, key=lambda p: p.n_changed)
