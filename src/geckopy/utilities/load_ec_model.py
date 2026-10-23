"""Load an ecModel from a YAML file.

Thin wrapper around :func:`raven_toolbox.io.read_yaml_model`. raven-toolbox
owns the YAML schema for the ec sections (see
``raven_toolbox.io.ec_data``): it reads the cobra-shaped portion plus the
``ec-rxns`` / ``ec-enzymes`` / ``gecko_light`` top-level sections,
populates ``model.ec`` as a typed :class:`~raven_toolbox.io.EcData`, and
normalises three legacy MATLAB GECKO quirks (top-level ``smiles`` on
metabolites, reverse-direction ``usage_prot_*`` / ``prot_pool_exchange``,
bare-``-`` document root).

geckopy keeps three application-level concerns:

1. **File-extension validation** — only ``.yml`` / ``.yaml`` is accepted.
   SBML ecModel I/O has been removed; raise a helpful ValueError if the
   caller passes ``.xml`` / ``.sbml``.
2. **Adapter-aware path resolution** — relative filenames resolve under
   ``<adapter.params.path>/models/<filename>``.
3. **Not-an-ecModel guard + adapter attachment** — fast-fail with a
   clear message if the YAML lacks the ec sections, and attach the
   adapter to ``model.adapter`` so downstream geckopy code finds it.

4. **ecModel format** — a file whose protein usage and pool reactions
   raven-toolbox turned forward from the GECKO 3 direction gives a
   warning that names :func:`~geckopy.upgrade_project`; an ecModel
   format (``gecko_format`` in the file's ``metaData``, see
   :func:`~geckopy.save_ec_model`) above 4 is an error. GECKO's
   ``loadEcModel`` does the same.

geckopy has no global default adapter: pass one explicitly, or use an
absolute filename.

Ported from GECKO MATLAB: src/geckomat/utilities/loadEcModel.m.
"""
from __future__ import annotations

import re
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from raven_toolbox.io import EcData, read_yaml_model

from ..ec_model.ec_model import EcModel
from ..upgrade import upgrade_ec_model

if TYPE_CHECKING:
    from ..adapter import ModelAdapter


_HOOK_LOCK = threading.Lock()  # warnings.showwarning is one global per process
_DEFAULT_FILENAME = "ecModel.yml"
_YAML_SUFFIXES = {".yml", ".yaml"}


def load_ec_model(
    filename: Optional[Union[str, Path]] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> EcModel:
    """Load an ecModel from a YAML file.

    Parameters
    ----------
    filename
        Path to the ecModel file. If ``None``, defaults to ``ecModel.yml``.
        Relative paths resolve against
        ``<adapter.params.path>/models/<filename>``; absolute paths are
        used as-is and ``adapter`` may be ``None``.
    adapter
        A ``ModelAdapter`` used to resolve a relative ``filename`` and
        attached to the loaded model as ``model.adapter``. Required when
        ``filename`` is relative.

    Returns
    -------
    EcModel
        The loaded ecModel with a populated ``model.ec``. The
        ``gecko_light`` flag, ec arrays, and coupling matrix are all
        read from the file.

    Raises
    ------
    ValueError
        If the extension isn't YAML, the filename is relative without an
        adapter, the YAML lacks the GECKO ``ec-rxns`` / ``ec-enzymes``
        sections (in which case it isn't an ecModel), or its ecModel
        format is above 4.
    FileNotFoundError
        If the resolved file doesn't exist.
    """
    path = _resolve_path(filename, adapter)

    # read_yaml_model turns GECKO 3 protein reactions forward, with a warning
    # that names how many. That warning is counted and replaced; every other
    # warning goes through the caller's filters and display unchanged.
    turned = 0
    show = warnings.showwarning

    def count_turned(message, category, filename, lineno, file=None, line=None):
        nonlocal turned
        m = re.search(r"reverse-direction convention for (\d+) protein", str(message))
        if m:
            turned += int(m.group(1))
        else:
            show(message, category, filename, lineno, file, line)

    with _HOOK_LOCK, warnings.catch_warnings():
        warnings.filterwarnings("always", message=r"ecModel uses the older reverse-direction convention")
        warnings.showwarning = count_turned
        cobra_model = read_yaml_model(path)
    meta = (cobra_model.notes or {}).get("metaData") or {}
    if _format_number(meta.get("gecko_format")) > 4:
        raise ValueError(
            f"{path} has ecModel format {meta['gecko_format']} (written by "
            f"{meta.get('generator', 'an unknown release')}); this geckopy reads "
            "format 4 and older. Update geckopy to load it."
        )
    ec = getattr(cobra_model, "ec", None)
    if not isinstance(ec, EcData) or ec.n_rxns == 0 or ec.n_enzymes == 0:
        raise ValueError(
            f"{path}: YAML lacks the GECKO `ec-rxns` and/or `ec-enzymes` "
            "top-level sections — this is not a geckopy ecModel YAML."
        )

    ec_model = EcModel.from_cobra(
        cobra_model, adapter=adapter, gecko_light=ec.gecko_light,
    )
    ec_model.ec = ec
    # raven-toolbox turns the reactions with a negative lower bound; GECKO
    # turns those whose stoichiometry has the GECKO 3 sign, which also
    # covers a usage reaction fixed to zero.
    turned += len(upgrade_ec_model(ec_model))
    if turned:
        warnings.warn(
            f"{path} has {turned} protein usage and pool reactions in the GECKO 3 direction; "
            "they are turned forward in the loaded model. upgrade_project upgrades the "
            "files of the project.",
            stacklevel=2,
        )
    return ec_model


def _format_number(value) -> float:
    """``gecko_format`` as a number; a value that is not a number counts as
    no format, as GECKO's ``loadEcModel`` reads it with ``str2double``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_path(
    filename: Optional[Union[str, Path]],
    adapter: Optional["ModelAdapter"],
) -> Path:
    if filename is None:
        filename = _DEFAULT_FILENAME
    path = Path(filename)
    if path.suffix.lower() not in _YAML_SUFFIXES:
        raise ValueError(
            "ecModels are loaded from YAML only (.yml/.yaml). Got: "
            f"{path.suffix!r}. SBML ecModel I/O has been removed."
        )
    if not path.is_absolute():
        if adapter is None:
            raise ValueError(
                "filename is relative; supply an `adapter` whose "
                "`params.path` will be used to resolve it under "
                "`<path>/models/<filename>`."
            )
        path = adapter.params.path / "models" / path
    if not path.is_file():
        raise FileNotFoundError(f"ecModel file not found: {path}")
    return path
