"""Shared helper to resolve which ``ModelAdapter`` a function should use.

Many geckopy functions need an adapter for organism parameters
(taxonomy id, biomass reaction, ...) or for paths to data files.
There are two ways a caller might supply one:

- pass it explicitly as an ``adapter=...`` keyword argument;
- attach it to the model as ``model.adapter`` (the common case
  after ``make_ec_model``, which sets it automatically).

``resolve_adapter`` consolidates the "explicit-arg-or-fallback-to-
``model.adapter``-or-raise" pattern that 30+ functions duplicated
inline, with inconsistent error messages.

Ported from GECKO MATLAB: src/model_adapter/ModelAdapterManager.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .adapter import ModelAdapter


def resolve_adapter(
    model,
    adapter: Optional["ModelAdapter"] = None,
    *,
    purpose: str = "this function",
) -> "ModelAdapter":
    """Return the adapter to use, falling back to ``model.adapter``.

    Parameters
    ----------
    model
        Any object that may have an ``.adapter`` attribute (a cobra
        Model, an EcModel, ...). ``getattr`` is used so plain
        ``cobra.Model`` doesn't trip on the missing attribute.
    adapter
        An explicitly-passed adapter. If non-None, returned as-is
        (takes priority over ``model.adapter``).
    purpose
        Short description of why the calling function needs an
        adapter; included in the error message when none can be
        resolved. Example: ``"to read params.enzyme_comp"``.

    Returns
    -------
    ModelAdapter
        Either ``adapter`` (if given) or ``model.adapter``.

    Raises
    ------
    ValueError
        If both ``adapter`` and ``model.adapter`` are ``None``. The
        message tells the caller how to fix it.
    """
    resolved = (
        adapter if adapter is not None else getattr(model, "adapter", None)
    )
    if resolved is None:
        raise ValueError(
            f"No ModelAdapter available ({purpose}). Either pass one "
            f"as the `adapter` argument or attach one to `model.adapter` "
            f"(typically via `model.adapter = "
            f"ModelAdapter.from_folder(...)`)."
        )
    return resolved


def resolve_param(
    model,
    explicit,
    attr: str,
    *,
    adapter: Optional["ModelAdapter"] = None,
    purpose: str = "this function",
):
    """Return an explicit value, else fall back to ``adapter.params.<attr>``.

    Lets a function take the single parameter it needs directly, only
    requiring a full ``ModelAdapter`` when the value is not supplied.

    Parameters
    ----------
    model
        Any object that may have an ``.adapter`` attribute, used as the
        fallback source when ``adapter`` is not given.
    explicit
        The caller-supplied value. If not ``None``, it is returned as-is
        and no adapter is consulted at all.
    attr
        Name of the attribute to read off ``adapter.params`` when
        ``explicit`` is ``None``.
    adapter
        An explicitly-passed adapter, used instead of ``model.adapter``
        when resolving the fallback value.
    purpose
        Short description of why the calling function needs the value;
        forwarded to :func:`resolve_adapter` for its error message.

    Returns
    -------
    ``explicit`` if given, otherwise ``getattr(adapter.params, attr)``.
    """
    if explicit is not None:
        return explicit
    resolved = resolve_adapter(model, adapter, purpose=purpose)
    return getattr(resolved.params, attr)
