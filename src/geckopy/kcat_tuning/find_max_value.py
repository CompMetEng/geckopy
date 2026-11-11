"""Find the maximum kcat (or SA-derived kcat) for a set of EC numbers.

Ported from GECKO 3 MATLAB ``findMaxValue.m``, which GECKO 4 removed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..databases.brenda_loader import BrendaData


_KCAT_LABEL = "K_cat"
_SA_LABEL = "SA*Mw"


def find_max_value(
    ec_string: str,
    brenda: "BrendaData",
) -> tuple[float, str, str]:
    """Find the maximum kinetic value across one or more EC numbers.

    Searches both ``brenda.kcat_max`` and ``brenda.sa_max`` for each EC token
    in ``ec_string`` and returns the overall maximum together with
    its organism and parameter type. Wildcards in EC tokens (``-``)
    trigger prefix matching: ``"1.1.1.-"`` matches any code starting
    with ``"1.1.1."``.

    Ported from GECKO 3 MATLAB ``findMaxValue.m``, which GECKO 4 removed.

    Parameters
    ----------
    ec_string
        Space-separated EC tokens. Each token may optionally start
        with ``"EC"`` (matching the MATLAB convention) and may
        contain ``-`` wildcards.
    brenda
        Pre-loaded BRENDA tables.

    Returns
    -------
    value
        The overall maximum kinetic value (1/s).
    organism
        The organism contributing the maximum.
    parameter
        Either ``"K_cat"`` (came from the kcat table) or
        ``"SA*Mw"`` (came from the SA-derived kcat table).

    If no EC matches anything in either table, returns
    ``(0.0, "", "")``.
    """
    tokens = ec_string.split()
    if not tokens:
        return 0.0, "", ""

    best_value = 0.0
    best_organism = ""
    best_parameter = ""

    for token in tokens:
        if token.upper().startswith("EC"):
            token = token[2:]
        if not token:
            continue

        # find_max_value is by definition asking for the maximum, so we
        # consult the max-aggregated snapshot view regardless of the
        # adapter's `kcat_aggregate_brenda` setting.
        kcat_value, kcat_org = _max_in_table(
            token, brenda.kcat_max, "kcat", "organism",
        )
        sa_value, sa_org = _max_in_table(
            token, brenda.sa_max, "kcat", "organism",
        )

        if kcat_value >= sa_value and kcat_value > 0:
            token_value, token_org, token_param = (
                kcat_value, kcat_org, _KCAT_LABEL,
            )
        elif sa_value > 0:
            token_value, token_org, token_param = (
                sa_value, sa_org, _SA_LABEL,
            )
        else:
            continue

        if token_value > best_value:
            best_value = token_value
            best_organism = token_org
            best_parameter = token_param

    return best_value, best_organism, best_parameter


def _max_in_table(
    token: str,
    table,
    value_col: str,
    organism_col: str,
) -> tuple[float, str]:
    """Return (max_value, organism) for the given EC token in the table.

    Wildcard tokens (containing ``-``) prefix-match against the
    table's ``ec_code`` column. Otherwise an exact case-insensitive
    match is used.
    """
    if table.empty:
        return 0.0, ""

    if "-" in token:
        prefix = token.split("-", 1)[0]
        if not prefix:
            mask = table["ec_code"].str.len() > 0
        else:
            mask = table["ec_code"].str.lower().str.startswith(prefix.lower())
    else:
        mask = table["ec_code"].str.lower() == token.lower()

    matched = table[mask]
    if matched.empty:
        return 0.0, ""
    idx = matched[value_col].idxmax()
    return float(matched.loc[idx, value_col]), str(matched.loc[idx, organism_col])
