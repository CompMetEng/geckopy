"""Load the conventional (non-ec) starting GEM for a project.

Every geckopy project has a "starting GEM" — a plain cobra model
without any GECKO additions. This is the model that
``make_ec_model`` extends into an ecModel.

This helper reads that file from the path set in the adapter
(``adapter.params.conv_gem``). The extension picks the reader:
``.yml``/``.yaml`` via raven-toolbox's ``read_yaml_model`` (which,
unlike cobra's reader, keeps RAVEN fields such as per-reaction
``eccodes``), ``.json`` via
``load_json_model``, ``.mat`` via ``load_matlab_model``, and
anything else via ``read_sbml_model``.

Ported from GECKO MATLAB:
src/geckomat/utilities/loadConventionalGEM.m.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cobra
from raven_toolbox.io import read_yaml_model
from raven_toolbox.utils.solver import set_raven_tolerance

from ..solver_support import fast_solver_while_building, switch_to_scipy

if TYPE_CHECKING:
    from ..adapter import ModelAdapter


_YAML_SUFFIXES = {".yml", ".yaml"}
_JSON_SUFFIXES = {".json"}
_MAT_SUFFIXES = {".mat"}


def load_conventional_gem(adapter: "ModelAdapter") -> cobra.Model:
    """Load the starting GEM file pointed to by ``adapter``.

    Parameters
    ----------
    adapter
        A ``ModelAdapter`` whose ``params.conv_gem`` field gives
        the path to the GEM file. The path is used as-is; the
        adapter's loader has already resolved it relative to the
        adapter folder.

    Returns
    -------
    cobra.Model
        The loaded model, with the solver tolerance set to RAVEN's 1e-9.
        Use it as the input to ``make_ec_model``.

    Raises
    ------
    FileNotFoundError
        If ``params.conv_gem`` doesn't point to an existing file.
    """
    path = Path(adapter.params.conv_gem)
    if not path.is_file():
        raise FileNotFoundError(f"conv_gem file not found: {path}")
    suffix = path.suffix.lower()
    # With the scipy solver as cobra's default the model is read on glpk and converted: optlang
    # grows the scipy matrix densely, which takes hours for a genome-scale model.
    with fast_solver_while_building() as on_fast_solver:
        if suffix in _YAML_SUFFIXES:
            model = read_yaml_model(path)
        elif suffix in _JSON_SUFFIXES:
            model = cobra.io.load_json_model(str(path))
        elif suffix in _MAT_SUFFIXES:
            model = cobra.io.load_matlab_model(str(path))
        else:
            model = cobra.io.read_sbml_model(str(path))
    if on_fast_solver:
        switch_to_scipy(model)
    return set_raven_tolerance(model)
