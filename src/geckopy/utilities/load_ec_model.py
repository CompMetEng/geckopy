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

4. **Tolerated input**: a file that the YAML reader rejects is read again
   after the two forms that GECKO's ``loadEcModel`` (through RAVEN's
   ``readYAMLmodel``) accepts are rewritten as valid YAML: ``enzymes: !!omap``
   with no entries, as earlier GECKO and RAVEN versions wrote it for a
   reaction without enzymes (current writers omit the ``enzymes`` entry, which
   is read as no enzymes), and a
   value in double quotes that holds unescaped double quotes, such as
   ``""text""``, which RAVEN 2 wrote for a value that has quotes of its own.
   The value loaded is the one MATLAB loads: the outer pair of quotes is
   removed and the quotes inside stay.

5. **ecModel format**: a file whose protein usage and pool reactions
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

import json
import re
import tempfile
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from raven_toolbox.io import EcData, read_yaml_model

from ..ec_model.ec_model import EcModel
from ..solver_support import fast_solver_while_building, switch_to_scipy
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
        # With the scipy solver as cobra's default the file is read on glpk and the
        # ecModel is converted: optlang grows the scipy matrix densely, which takes
        # hours for a genome-scale model.
        with fast_solver_while_building() as on_fast_solver:
            cobra_model = _read_yaml(path)
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
    if on_fast_solver:
        switch_to_scipy(ec_model)
    # raven-toolbox turns the reactions whose stoichiometry has the GECKO 3
    # sign; the same rule is applied here to a model that was not read from
    # YAML.
    turned += len(upgrade_ec_model(ec_model))
    if turned:
        warnings.warn(
            f"{path} has {turned} protein usage and pool reactions in the GECKO 3 direction; "
            "they are turned forward in the loaded model. upgrade_project upgrades the "
            "files of the project.",
            stacklevel=2,
        )
    return ec_model


_EMPTY_OMAP = re.compile(
    # A key whose value is the tag `!!omap` and nothing else, when the next
    # non-blank line is not indented deeper than the line of the key.
    r"^( *)((?:- )?[^\n:]*: )!!omap[ \t]*$(?!\n(?:[ \t]*\n)*\1 +\S)",
    re.MULTILINE,
)
_QUOTED_VALUE = re.compile(
    r'^( *-(?: +[^\s":][^":\n]*:)? +)"(.*)"[ \t]*$',
    re.MULTILINE,
)


def _read_yaml(path: Path):
    """``read_yaml_model``; a file it rejects is read again after
    :func:`_repair_yaml_text`, and the first error is raised when that
    changes nothing."""
    try:
        return read_yaml_model(path)
    except Exception:
        text = path.read_text(encoding="utf-8")
        repaired = _repair_yaml_text(text)
        if repaired == text:
            raise
    with tempfile.TemporaryDirectory() as folder:
        fixed = Path(folder) / "ecModel.yml"
        fixed.write_text(repaired, encoding="utf-8")
        return read_yaml_model(fixed)


def _repair_yaml_text(text: str) -> str:
    """Write the two forms that MATLAB reads and YAML does not as YAML.

    ``key: !!omap`` without entries becomes ``key: !!omap []``. A value in
    double quotes that holds a double quote which is not escaped is read as
    RAVEN's ``readYAMLmodel`` reads it: without its outer pair of quotes and
    with an escaped backslash or quote as the character itself, and written
    back as a YAML string.
    """
    text = _EMPTY_OMAP.sub(r"\1\2!!omap []", text)

    def requote(match: re.Match) -> str:
        inner = match.group(2)
        if not _has_unescaped_quote(inner):
            return match.group(0)
        value = inner.replace("\\\\", "\0").replace('\\"', '"').replace("\0", "\\")
        return match.group(1) + json.dumps(value, ensure_ascii=False)

    return _QUOTED_VALUE.sub(requote, text)


def _has_unescaped_quote(inner: str) -> bool:
    """Whether ``inner``, the text between the outer quotes of a value, has a
    double quote that no backslash escapes."""
    i = 0
    while i < len(inner):
        if inner[i] == "\\":
            i += 2
            continue
        if inner[i] == '"':
            return True
        i += 1
    return False


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
