"""Fit the average enzyme saturation factor (``sigma``).

The protein pool's upper bound is set to ``P_tot * f * sigma``,
where ``sigma`` is the average enzyme saturation factor — a
fudge factor between 0 and 1 capturing the fact that enzymes
don't usually run at their full Vmax in vivo. The default value
is 0.5; this function fits a better one.

The algorithm: try a range of sigma values, set the protein-pool
bound at each, solve the model, compare the predicted growth
rate to the experimental one, pick the sigma that minimises the
difference. The default scan is the 0.01 grid of GECKO's
``sigmaFitter``; the objective reaction's bounds are opened to
``[0, 1000]`` first, and stay open on the fitted model.

Ported from GECKO MATLAB:
src/geckomat/kcat_sensitivity_analysis/sigmaFitter.m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from ..ec_model.pipeline.protein_pool import set_prot_pool_size

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

# Sigma resolution the bisection converges to (fine enough to resolve a steep
# growth curve; ~20 iterations from the [0, 1] bracket).
_SIGMA_BISECT_TOL = 1e-6
# Relative tolerance within which growth counts as being on its plateau, so
# solver noise does not move the start of the plateau.
_PLATEAU_RTOL = 1e-9


@dataclass
class SigmaFitterResult:
    """Outcome of a sigma scan.

    Attributes
    ----------
    sigma
        The sigma value that minimised the absolute relative error
        between predicted and experimental growth.
    sigma_grid
        Sigma values that were tried.
    growth_grid
        LP-optimal growth at each sigma.
    error_grid
        Absolute relative error at each sigma, in percent.
    """

    sigma: float = 0.0
    sigma_grid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    growth_grid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    error_grid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


def _growth_at_sigma(
    model: "EcModel", p_tot: float, f: float, sigma: float,
) -> float:
    """Solve the model with the pool sized for ``sigma``; revert after.

    A solve that is not optimal counts as growth 0, as an empty solution
    does in ``sigmaFitter.m``.
    """
    with model:
        set_prot_pool_size(model, p_tot=p_tot, f=f, sigma=float(sigma))
        sol = model.optimize()
        if sol.status != "optimal" or sol.objective_value is None:
            return 0.0
        value = float(sol.objective_value)
        return value if np.isfinite(value) else 0.0


def _sigma_error_pct(growth: float, growth_rate: float) -> float:
    if growth_rate != 0:
        return abs((growth_rate - growth) / growth_rate) * 100.0
    return abs(growth) * 100.0


def fit_sigma(
    model: "EcModel",
    *,
    growth_rate: Optional[float] = None,
    p_tot: Optional[float] = None,
    f: Optional[float] = None,
    n_sigma_steps: int = 100,
    method: Literal["grid", "bisect"] = "grid",
) -> SigmaFitterResult:
    """Sweep sigma and pick the one matching ``growth_rate`` best.

    ``method="grid"`` (default) is the sweep of GECKO's ``sigmaFitter``:
    sigma ``i / n`` for ``i = 1..n`` (every grid point evaluated), and the
    first grid point with the smallest error wins. For a target that growth
    cannot reach this is the first sigma of the growth plateau when the
    plateau values are identical; values that differ by solver noise can move
    the pick to a later grid point, as in ``sigmaFitter.m``. A model that
    cannot grow at any sigma gives the first grid point, ``1 / n_sigma_steps``.
    ``method="bisect"`` exploits that growth is monotone non-decreasing in
    sigma and finds the best sigma, to a resolution of 1e-6 and not on the
    grid, in about ``log2(n)`` solves; the returned grids then contain only
    the sigmas actually evaluated. For an unreachable target it finds where
    the growth plateau starts (growth within a relative 1e-9 of its value at
    sigma 1); a model that cannot grow at sigma 1 gives ``1 / n_sigma_steps``,
    as the grid does.

    The bounds of the objective reaction are set to ``[0, 1000]`` before the
    scan, as ``sigmaFitter.m`` does, and stay so on ``model``. The best sigma
    is re-applied to ``model`` before return.

    Ported from GECKO MATLAB:
    src/geckomat/kcat_sensitivity_analysis/sigmaFitter.m.

    Parameters
    ----------
    model
        EcModel with the protein pool machinery installed and the
        objective set (typically the biomass reaction). Mutated in
        place: the objective reaction's bounds become ``[0, 1000]`` and
        the optimal sigma is applied via ``set_prot_pool_size`` before
        return.
    growth_rate
        Experimental growth rate to match. Defaults to
        ``model.adapter.params.gr_exp``.
    p_tot
        Total protein content (g/gDCW). Defaults to
        ``model.adapter.params.p_tot``.
    f
        Mass fraction of model enzymes. Defaults to
        ``model.adapter.params.f``.
    n_sigma_steps
        For ``method="grid"``, the number of sigma values tried in
        ``(0, 1]`` (``i / n`` for ``i = 1..n``). For ``method="bisect"``,
        it caps the number of bisection iterations (the default 100 is far
        more than the ~20 needed to converge). Default 100.
    method
        ``"grid"`` (default) sweeps the full grid as ``sigmaFitter.m``
        does; ``"bisect"`` finds the best sigma in ~``log2(n)`` solves using
        the monotonicity of growth in sigma.

    Returns
    -------
    SigmaFitterResult

    Raises
    ------
    ValueError
        If ``model.adapter`` is None and any default is needed; if
        ``n_sigma_steps`` is < 1; or if ``method`` is unknown.
    """
    if n_sigma_steps < 1:
        raise ValueError(f"n_sigma_steps must be >= 1, got {n_sigma_steps}")
    if method not in ("grid", "bisect"):
        raise ValueError(f"method must be 'grid' or 'bisect', got {method!r}")

    if (
        (growth_rate is None or p_tot is None or f is None)
        and model.adapter is None
    ):
        raise ValueError(
            "model.adapter is None and one of growth_rate / p_tot / f "
            "is not provided."
        )
    params = model.adapter.params if model.adapter is not None else None

    # sigmaFitter.m opens the bounds of the objective for good.
    for rxn in model.reactions:
        if rxn.objective_coefficient != 0:
            rxn.bounds = (0.0, 1000.0)

    if growth_rate is None:
        growth_rate = float(params.gr_exp)
    if p_tot is None:
        p_tot = float(params.p_tot)
    if f is None:
        f = float(params.f)

    plateau_sigma: Optional[float] = None
    if method == "grid":
        sigma_grid = np.array(
            [(i + 1) / n_sigma_steps for i in range(n_sigma_steps)],
            dtype=float,
        )
        growth_grid = np.array(
            [_growth_at_sigma(model, p_tot, f, s) for s in sigma_grid],
            dtype=float,
        )
    else:  # "bisect": growth is monotone non-decreasing in sigma.
        evaluated: dict[float, float] = {}

        def growth_at(sigma: float) -> float:
            sigma = float(sigma)
            if sigma not in evaluated:
                evaluated[sigma] = _growth_at_sigma(model, p_tot, f, sigma)
            return evaluated[sigma]

        g_max = growth_at(1.0)
        if g_max < growth_rate:
            # Target unreachable even at the full budget: the closest growth
            # is the plateau, so look for the smallest sigma that reaches it
            # (to solver noise).
            reach = g_max - _PLATEAU_RTOL * abs(g_max)
        else:
            reach = growth_rate
        # Bisect for the smallest sigma reaching ``reach``. The tolerance is
        # fine (not 1/n) so a steep growth curve still resolves the
        # crossing; n_sigma_steps just caps iterations.
        lo, hi = 0.0, 1.0
        if g_max <= 0:
            # No growth at any sigma: every error is equal, and the grid takes
            # its first point.
            hi = 1.0 / n_sigma_steps
        else:
            for _ in range(n_sigma_steps):
                if hi - lo <= _SIGMA_BISECT_TOL:
                    break
                mid = 0.5 * (lo + hi)
                if growth_at(mid) >= reach:
                    hi = mid
                else:
                    lo = mid
        best_sigma = hi
        if g_max < growth_rate:
            plateau_sigma = hi
        growth_at(best_sigma)
        sigma_grid = np.array(sorted(evaluated), dtype=float)
        growth_grid = np.array(
            [evaluated[s] for s in sigma_grid], dtype=float,
        )

    error_grid = np.array(
        [_sigma_error_pct(g, growth_rate) for g in growth_grid], dtype=float,
    )
    if plateau_sigma is not None:
        best_sigma = float(plateau_sigma)
    else:
        best_sigma = float(sigma_grid[int(np.argmin(error_grid))])

    # Apply the best sigma to the model permanently (geckopy divergence).
    set_prot_pool_size(model, p_tot=p_tot, f=f, sigma=best_sigma)

    return SigmaFitterResult(
        sigma=best_sigma,
        sigma_grid=sigma_grid,
        growth_grid=growth_grid,
        error_grid=error_grid,
    )


def sigma_fitter(
    model: "EcModel",
    *,
    growth_rate: Optional[float] = None,
    p_tot: Optional[float] = None,
    f: Optional[float] = None,
    n_sigma_steps: int = 100,
) -> SigmaFitterResult:
    """Deprecated alias for :func:`fit_sigma`.

    Kept for backward compatibility. Will be removed in a future
    release; switch to ``fit_sigma``.
    """
    import warnings

    warnings.warn(
        "sigma_fitter is deprecated; use fit_sigma instead. "
        "The old name will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_sigma(
        model,
        growth_rate=growth_rate, p_tot=p_tot, f=f,
        n_sigma_steps=n_sigma_steps,
    )
