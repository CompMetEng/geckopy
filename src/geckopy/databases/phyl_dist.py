"""Load KEGG phylogenetic distance data from GECKO MATLAB's PhylDist.mat.

Used by fuzzy_kcat_matching to find the evolutionarily closest organism
when an exact match is not available in BRENDA.

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/fuzzyKcatMatching.m (the inline `KEGG_struct` helper).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat


# keggPhylDist.mat as distributed with RAVEN, pinned to a release tag so
# the download is reproducible.
KEGG_PHYL_DIST_URL = (
    "https://raw.githubusercontent.com/CompMetEng/RAVEN/3.0.0b1/"
    "reconstruction/kegg/keggPhylDist.mat"
)
_TIMEOUT_SECONDS = 120

# Strip a trailing parenthetical comment, e.g.
# `"Saccharomyces cerevisiae (baker's yeast)"` -> `"Saccharomyces cerevisiae"`.
_PAREN_TAIL_RE = re.compile(r"\s*\(.*$")


@dataclass
class PhylDist:
    """KEGG phylogenetic distance structure.

    Attributes
    ----------
    names
        KEGG organism names, with any trailing parenthetical comment
        stripped. Stored in original case.
    dist_matrix
        ``N x N`` pairwise evolutionary distance matrix.
        ``dist_matrix[i, j]`` is the distance between ``names[i]``
        and ``names[j]``.
    name_to_index
        Lowercased name -> index in ``names``. First occurrence wins
        for duplicate names (matching MATLAB ``find(..., 1)``).
    genus_to_indices
        Lowercased genus (first whitespace-delimited word of the
        name) -> list of indices in ``names`` whose genus matches.
    """

    names: list[str] = field(default_factory=list)
    dist_matrix: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    name_to_index: dict[str, int] = field(default_factory=dict)
    genus_to_indices: dict[str, list[int]] = field(default_factory=dict)


def load_phyl_dist(path: str | Path) -> PhylDist:
    """Load a KEGG phylogenetic distance struct from a MATLAB .mat file.

    The .mat file is expected to contain a single variable
    ``phylDistStruct`` with fields ``names`` (cell of strings) and
    ``distMat`` (``N x N`` numeric matrix). The MATLAB ``ids`` field
    is ignored: ``fuzzy_kcat_matching`` never uses it.

    Names are stripped of any trailing parenthetical comment, matching
    MATLAB's ``regexprep(names, '\\s*\\(.*', '')``. The resulting
    cleaned names populate two pre-computed lookup maps for fast
    organism resolution at match time:

    * ``name_to_index``: lowercased full name -> first matching index.
    * ``genus_to_indices``: lowercased genus (first whitespace token)
      -> all matching indices.

    Ported from GECKO MATLAB (inline ``KEGG_struct`` helper in
    src/geckomat/gather_kcats/fuzzyKcatMatching.m).

    Parameters
    ----------
    path
        Path to ``PhylDist.mat``.

    Returns
    -------
    PhylDist
        Populated dataclass.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If the .mat file does not contain a ``phylDistStruct``
        variable with both ``names`` and ``distMat`` fields.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PhylDist.mat not found: {path}")

    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "phylDistStruct" not in raw:
        raise KeyError(
            f"{path}: expected variable 'phylDistStruct' in .mat file."
        )
    struct = raw["phylDistStruct"]

    if not (hasattr(struct, "names") and hasattr(struct, "distMat")):
        raise KeyError(
            f"{path}: phylDistStruct missing 'names' or 'distMat' field."
        )

    names = [
        _clean_name(str(n)) for n in np.asarray(struct.names).ravel()
    ]
    dist_mat = np.asarray(struct.distMat, dtype=float)

    name_to_index: dict[str, int] = {}
    genus_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        lower = name.lower()
        name_to_index.setdefault(lower, i)
        parts = lower.split(None, 1)
        if parts:
            genus_to_indices.setdefault(parts[0], []).append(i)

    return PhylDist(
        names=names,
        dist_matrix=dist_mat,
        name_to_index=name_to_index,
        genus_to_indices=genus_to_indices,
    )


def _clean_name(name: str) -> str:
    """Strip a trailing parenthetical comment and trim whitespace."""
    return _PAREN_TAIL_RE.sub("", name).strip()


def download_phyl_dist(
    out_path: str | Path,
    *,
    url: str = KEGG_PHYL_DIST_URL,
    session: requests.Session | None = None,
) -> Path:
    """Download the KEGG phylogenetic distance file distributed with RAVEN.

    ``fuzzy_kcat_matching`` uses it to pick the evolutionarily closest
    organism when BRENDA has no entry for the model's own organism.
    The file covers all KEGG organisms (~5 MB). It is written to a
    temporary file next to ``out_path`` and only moved into place once
    ``load_phyl_dist`` can read it, so a failed download never leaves
    a partial file at ``out_path``.

    Parameters
    ----------
    out_path
        Destination ``.mat`` path (overwritten if it exists), typically
        ``adapter.get_phyl_dist_path()``.
    url
        Source URL. Defaults to ``keggPhylDist.mat`` of RAVEN 3.0.0b1.
    session
        Optional requests.Session for connection reuse and test
        injection. A new transient session is created if absent.

    Returns
    -------
    Path
        The written ``out_path``.

    Raises
    ------
    requests.HTTPError
        If the download fails.
    KeyError
        If the downloaded file is not a ``phylDistStruct`` .mat file.
    """
    out_path = Path(out_path)
    sess = session or requests.Session()
    resp = sess.get(url, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".part")
    try:
        tmp_path.write_bytes(resp.content)
        load_phyl_dist(tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path
