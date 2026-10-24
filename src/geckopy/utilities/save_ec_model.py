"""Save an ecModel to a YAML file.

Thin wrapper around :func:`raven_toolbox.io.write_yaml_model`. raven-toolbox
owns the YAML schema for the ec sections (see
``raven_toolbox.io.ec_data``): given a ``model`` with a populated
``model.ec`` (an :class:`~raven_toolbox.io.EcData`), it serialises both the
cobra-shaped portion and the ``ec-rxns`` / ``ec-enzymes`` / ``gecko_light``
top-level sections.

geckopy keeps three application-level concerns:

1. **File-extension validation** — only ``.yml`` / ``.yaml`` is accepted.
   SBML ecModel I/O has been removed; raise a helpful ValueError if the
   caller passes ``.xml`` / ``.sbml``.
2. **Adapter-aware path resolution** — relative filenames resolve under
   ``<adapter.params.path>/models/<filename>``.
3. **Empty-ecModel guard + provenance injection** — fast-fail when the
   caller forgot to run ``make_ec_model``; write a small ``metaData``
   block (date, ecModel format, geckopy release, description) so saved
   files carry provenance information.

The provenance/description metadata is written only into the saved
YAML; the in-memory model's ``notes`` are mutated only transiently for
the duration of the write (provenance stashed there so raven-toolbox's
writer emits it) and are restored afterwards, leaving the caller's
model unchanged.

Ported from GECKO MATLAB: src/geckomat/utilities/saveEcModel.m.
"""
from __future__ import annotations

from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from raven_toolbox.io import write_yaml_model

from ..ec_model.ec_model import EcModel

if TYPE_CHECKING:
    from ..adapter import ModelAdapter


_DEFAULT_FILENAME = "ecModel.yml"
_YAML_SUFFIXES = {".yml", ".yaml"}


def save_ec_model(
    model: EcModel,
    filename: Optional[Union[str, Path]] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> Path:
    """Save an ecModel to a YAML file.

    Parameters
    ----------
    model
        The ecModel to save. Must have a populated ``model.ec`` (both
        ``ec.rxns`` and ``ec.enzymes`` non-empty). Empty ecModels are
        rejected because they're almost always a bug — you probably
        forgot to run ``make_ec_model`` first.
    filename
        Destination path. If ``None``, defaults to ``ecModel.yml``.
        Relative paths resolve against
        ``<adapter.params.path>/models/<filename>``; absolute paths are
        used as-is.
    adapter
        Used to resolve a relative ``filename``. Defaults to
        ``model.adapter``. Required when ``filename`` is relative AND
        ``model.adapter`` is ``None``.

    Returns
    -------
    Path
        The absolute path of the file that was written.

    Raises
    ------
    ValueError
        If ``model.ec`` is empty, if the file extension isn't YAML, or
        if the filename is relative and no adapter is available.
    """
    if model.ec is None or len(model.ec.rxns) == 0 or len(model.ec.enzymes) == 0:
        raise ValueError(
            "model.ec is empty (zero rxns or zero enzymes); save_ec_model "
            "requires a populated ecModel. Run `make_ec_model` first."
        )

    resolved_adapter = adapter if adapter is not None else getattr(
        model, "adapter", None,
    )
    path = _resolve_path(filename, resolved_adapter)

    # Inject provenance into model.notes for the duration of the write so
    # raven-toolbox's writer emits a metaData block at top level. Restore
    # the caller's notes in a finally block so the in-memory model isn't
    # permanently altered.
    saved_notes = dict(model.notes or {})
    try:
        model.notes["metaData"] = _build_metadata(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml_model(model, path)
    finally:
        model.notes = saved_notes

    return path


def _resolve_path(
    filename: Optional[Union[str, Path]],
    adapter: Optional["ModelAdapter"],
) -> Path:
    if filename is None:
        filename = _DEFAULT_FILENAME
    path = Path(filename)
    if path.suffix.lower() not in _YAML_SUFFIXES:
        raise ValueError(
            "ecModels are saved as YAML only (.yml/.yaml). Got: "
            f"{path.suffix!r}. SBML ecModel I/O has been removed."
        )
    if not path.is_absolute():
        if adapter is None:
            raise ValueError(
                "filename is relative; supply an `adapter` (or set "
                "`model.adapter`) whose `params.path` will be used to "
                "resolve it under `<path>/models/<filename>`."
            )
        path = adapter.params.path / "models" / path
    return path


def _build_metadata(model: EcModel) -> dict:
    """Provenance block written under the top-level ``metaData`` key.

    The model's own ``metaData`` (from the file it was read from, or the
    conventional model it was made from) with ``date`` kept or set to
    today, as RAVEN keeps ``model.date``; ``gecko_format``, the ecModel
    format the file follows (as ``project_format`` in
    ``model_adapter.toml``, written as text as GECKO's ``saveEcModel``
    writes it); ``generator``, the release that wrote the file; and a
    description string of the form ``"Enzyme-constrained model of <id>"``.
    GECKO's ``saveEcModel`` writes the same ``gecko_format`` and
    ``generator``.
    """
    try:
        geckopy_version = version("geckopy")
    except PackageNotFoundError:
        geckopy_version = "unknown"
    meta = dict((model.notes or {}).get("metaData") or {})
    meta.pop("geckopy_version", None)
    meta["date"] = meta.get("date") or date.today().isoformat()
    meta["gecko_format"] = "4"
    meta["generator"] = f"geckopy {geckopy_version}"
    meta["description"] = f"Enzyme-constrained model of {model.id}"
    return meta
