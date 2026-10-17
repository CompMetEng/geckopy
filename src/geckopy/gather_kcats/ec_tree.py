"""The EC tree: an EC number's median kcat shrunk toward its parent EC classes.

For each EC number, the median log10 kcat of its measurements is pulled
toward the estimate of its parent class with weight ``n / (n + k)``, where
``n`` is the number of measurements: the global median, then the 2-digit
class, the 3-digit class and the full EC number, in that order. A
well-measured EC keeps its own median; a sparsely measured one leans on its
class. GECKO has the same in ``kcatsFromECTree``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _parent(ec: str, depth: int) -> str:
    return ".".join(ec.split(".")[:depth])


class ECTree:
    """EC-tree estimates from a BRENDA measurement table (``ec``, ``log10_kcat``)."""

    def __init__(self, measurements: pd.DataFrame, k: float = 5.0):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = float(k)
        ec = measurements["ec"].astype(str)
        logk = measurements["log10_kcat"].to_numpy(dtype=float)
        self.global_median = float(np.median(logk))
        self.levels = {}
        for depth in (2, 3, 4):
            key = ec if depth == 4 else ec.map(lambda e, d=depth: _parent(e, d))
            g = pd.Series(logk).groupby(key.to_numpy())
            self.levels[depth] = (g.median().to_dict(), g.size().to_dict())

    def estimate(self, eccodes: str) -> tuple[float, int, int, str]:
        """log10 kcat, measurements of the full EC, depth of the deepest level
        with measurements (0, 2, 3 or 4) and the EC number used.

        ``eccodes`` holds one or more EC numbers separated by ``;``; the one
        with most measurements of its full EC is used (the first on a tie).
        Without an EC number the depth is 0 and the value the global median.
        """
        ecs = [e.strip() for e in str(eccodes or "").split(";") if e.strip()]
        if not ecs:
            return self.global_median, 0, 0, ""
        best, best_n = ecs[0], -1
        for ec in ecs:
            n = self.levels[4][1].get(ec, 0)
            if n > best_n:
                best, best_n = ec, n
        value, depth = self.global_median, 0
        for d in (2, 3, 4):
            key = best if d == 4 else _parent(best, d)
            n = self.levels[d][1].get(key, 0)
            if n:
                w = n / (n + self.k)
                value = w * self.levels[d][0][key] + (1 - w) * value
                depth = d
        return value, best_n, depth, best
