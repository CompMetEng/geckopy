"""Round numeric values to roughly 6 significant figures.

Ported from GECKO MATLAB:
src/geckomat/kcat_sensitivity_analysis/truncateValues.m.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

import numpy as np


def truncate_values(
    arr: np.ndarray,
    columns: Optional[list[int]] = None,
) -> np.ndarray:
    """Round values in ``arr`` to ~6 significant figures.

    For each value ``v``:

        order_magn = max(ceil(log10(|v|)), 0)
        result     = round(v, 6 - order_magn)

    A value exactly halfway between two candidates rounds away from
    zero, as MATLAB's ``round(x, n)`` does (``100000.5`` gives
    ``100001`` and ``1234565`` gives ``1234570``). The value is taken
    at its shortest decimal representation, so ``2.675`` is a tie.
    Zero, NaN, and infinite values are returned unchanged.

    For 2D input, only the columns selected by ``columns`` are
    processed (default: all). Other columns are copied through
    unchanged. The input is never mutated; a new array is returned.

    Ported from GECKO MATLAB:
    src/geckomat/kcat_sensitivity_analysis/truncateValues.m.

    Parameters
    ----------
    arr
        1-D or 2-D numeric array.
    columns
        Column indices to process for 2-D input. Default: every
        column. Ignored for 1-D input.

    Returns
    -------
    numpy.ndarray
        New array with the truncated values.
    """
    out = np.asarray(arr, dtype=float).copy()

    if out.ndim == 1:
        for i in range(out.shape[0]):
            out[i] = _truncate_scalar(out[i])
        return out

    if out.ndim != 2:
        raise ValueError(
            f"arr must be 1-D or 2-D; got {out.ndim}-D"
        )

    if columns is None:
        cols_to_process = range(out.shape[1])
    else:
        cols_to_process = columns

    for j in cols_to_process:
        for i in range(out.shape[0]):
            out[i, j] = _truncate_scalar(out[i, j])
    return out


def _truncate_scalar(v: float) -> float:
    """Round ``v`` to ~6 significant figures.

    Ties round away from zero. Zero, NaN, and infinite values are
    returned unchanged.
    """
    if v == 0 or np.isnan(v) or np.isinf(v):
        return v
    order_magn = max(int(np.ceil(np.log10(abs(v)))), 0)
    quantum = Decimal(1).scaleb(order_magn - 6)
    return float(Decimal(repr(float(v))).quantize(quantum, rounding=ROUND_HALF_UP))
