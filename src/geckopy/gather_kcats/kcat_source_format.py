"""Write and read the provenance strings of ``model.ec.source``.

A string is ``<origin>[ (<detail>)][ | tuned=<method>[+<method>...] (prior=<value>)]``,
for example ``"brenda (wc=0, origin=1)"`` or
``"okp_catapro (similarity=42) | tuned=sensitivity+cmaes (prior=5.48)"``.
The origin names where the kcat came from, the detail holds ``key=value``
pairs of that origin, and the tuning status lists the tuning methods applied
to the kcat in order together with the kcat before the first of them.

Mirrors GECKO MATLAB's ``formatKcatSource`` and ``parseKcatSource``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable, Mapping, Optional

from .merge_kcats import normalize_source

_TUNED_MARK = " | tuned="
_NUMBER = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
_PARSE_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FORMAT_KEY = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class KcatSource:
    """The parts of one ``ec.source`` string.

    Attributes
    ----------
    origin
        The origin as written (case is not changed); ``""`` for a missing
        string.
    detail
        The ``key=value`` pairs of the detail: a float when the value
        parses as a number, a str otherwise.
    tuned
        The tuning methods in order of application.
    prior
        The kcat before the first tuning, or None.
    """

    origin: str
    detail: dict[str, float | str] = field(default_factory=dict)
    tuned: tuple[str, ...] = ()
    prior: Optional[float] = None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def _number_text(value: float) -> str:
    return f"{float(value):.6g}"


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and math.isfinite(float(value))


def format_kcat_source(
    origin: object,
    detail: Optional[Mapping[str, object]] = None,
    tuned: Iterable[str] = (),
    prior: Optional[float] = None,
) -> str:
    """Build the provenance string written to ``model.ec.source``.

    Mirrors GECKO MATLAB's ``formatKcatSource``.

    Parameters
    ----------
    origin
        Where the kcat came from (``"brenda"``, ``"custom"``,
        ``"okp_catapro"``, ...). Lower-cased, with every run of characters
        outside ``[0-9a-z]`` folded to one ``_`` (``"OKP-CataPro"`` gives
        ``"okp_catapro"``).
    detail
        ``key=value`` pairs written in the order of the mapping, keys
        normalised as the origin. A number is written with ``%.6g``, text
        verbatim. A pair whose value is None, NaN or infinite is left out,
        so is a pair whose key does not start with a letter after the
        normalisation, and the brackets are left out when no pair is left.
    tuned
        Tuning methods applied to the kcat, in order of application. The
        tuning status is written when this is not empty.
    prior
        The kcat before the first tuning, written as ``(prior=<value>)``
        after the methods; None or NaN means none.

    Returns
    -------
    str
        The provenance string. An empty ``origin`` gives ``""`` when there
        is no detail and no tuning status, and the origin ``unknown``
        otherwise.

    Examples
    --------
    >>> format_kcat_source("brenda", {"wc": 0, "origin": 1})
    'brenda (wc=0, origin=1)'
    >>> format_kcat_source("OKP-CataPro", {"similarity": 42}, ["cmaes"], 5.48)
    'okp_catapro (similarity=42) | tuned=cmaes (prior=5.48)'
    """
    token = normalize_source(origin)
    pairs = []
    for key, value in (detail or {}).items():
        name = normalize_source(key)
        if not _FORMAT_KEY.fullmatch(name) or _is_missing(value):
            continue
        if isinstance(value, Real):
            if not math.isfinite(float(value)):
                continue
            text = _number_text(value)
        else:
            text = str(value)
        pairs.append(f"{name}={text}")
    if tuned is None:
        tuned = ()
    methods = [str(m) for m in ([tuned] if isinstance(tuned, str) else tuned)]
    if not token and not pairs and not methods:
        return ""
    text = token or "unknown"
    if pairs:
        text += f" ({', '.join(pairs)})"
    if methods:
        text += _TUNED_MARK + "+".join(methods)
        if _is_number(prior):
            text += f" (prior={_number_text(prior)})"
    return text


def _parse_value(text: str) -> float | str:
    return float(text) if _NUMBER.fullmatch(text) else text


def parse_kcat_source(text: object) -> KcatSource:
    """Split a provenance string of ``model.ec.source`` into its parts.

    Mirrors GECKO MATLAB's ``parseKcatSource``. A string that does not
    follow the format is read as an origin with no detail and no tuning
    status, so a string written by an earlier version stays readable.

    Parameters
    ----------
    text
        One ``ec.source`` entry. None, NaN and anything that is not a str
        give the origin ``""``.

    Returns
    -------
    KcatSource
        ``origin`` as written, ``detail`` (numbers as float), ``tuned``
        (methods in order) and ``prior`` (None when absent).

    Examples
    --------
    >>> info = parse_kcat_source("brenda (wc=0, origin=2) | tuned=cmaes (prior=6.55)")
    >>> info.origin, info.detail, info.tuned, info.prior
    ('brenda', {'wc': 0.0, 'origin': 2.0}, ('cmaes',), 6.55)
    """
    if not isinstance(text, str):
        return KcatSource("")
    head, mark, tail = text.strip().partition(_TUNED_MARK)
    head = head.strip()
    origin, bracket, rest = head.partition(" (")
    detail: dict[str, float | str] = {}
    if bracket and rest.endswith(")"):
        for part in rest[:-1].split(","):
            key, equals, value = part.partition("=")
            if equals and _PARSE_KEY.fullmatch(key.strip()):
                detail[key.strip()] = _parse_value(value.strip())
    elif bracket:
        origin = head
    methods: tuple[str, ...] = ()
    prior: Optional[float] = None
    if mark:
        tail = tail.strip()
        names, bracket, rest = tail.partition(" (prior=")
        if (
            bracket
            and rest.endswith(")")
            and ")" not in rest[:-1]
            and not names[-1:].isspace()
        ):
            value = _parse_value(rest[:-1].strip())
            prior = value if isinstance(value, float) else None
        else:
            names = tail
        methods = tuple(m.strip() for m in names.split("+") if m.strip())
    return KcatSource(origin.strip(), detail, methods, prior)


def is_standard_source(source: object) -> bool:
    """Whether an ``ec.source`` entry marks the standard kcat.

    True when the origin is ``standard`` (compared exactly) and the string has
    no tuning status, so a tuned standard kcat counts as set on purpose.
    """
    info = parse_kcat_source(source)
    return info.origin == "standard" and not info.tuned
