"""EC-number wildcard escalation.

When a BRENDA search misses, the matcher loosens the EC token by
replacing its rightmost numeric level with ``-`` and trying
again. These helpers do just that string manipulation -- no
BRENDA, no model.

Ported from GECKO MATLAB: src/gather_kcats/fuzzyKcatMatching.m.
"""
from __future__ import annotations

from typing import Optional


def escalate_wildcard(ec_token: str) -> Optional[str]:
    """Replace the rightmost numeric level of ``ec_token`` with ``-``.

    Returns the new token, or ``None`` if every level is already
    ``-`` (fully wildcarded).
    """
    parts = ec_token.split(".")
    if len(parts) != 4:
        return None
    for i in range(3, -1, -1):
        if parts[i] != "-":
            parts[i] = "-"
            return ".".join(parts)
    return None


def apply_force_wildcards(ec_token: str, force_level: int) -> str:
    """Escalate ``ec_token`` by ``force_level`` wildcards from the right.

    A no-op when ``force_level == 0`` (the common case).
    """
    for _ in range(force_level):
        nxt = escalate_wildcard(ec_token)
        if nxt is None:
            return ec_token
        ec_token = nxt
    return ec_token
