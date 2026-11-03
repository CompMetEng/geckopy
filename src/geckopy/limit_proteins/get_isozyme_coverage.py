"""Report isozyme groups with partial proteomics coverage.

Ported from GECKO MATLAB:
src/geckomat/limit_proteins/getIsozymeCoverage.m.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

_COLUMNS = [
    "rxn", "n_entries", "n_measured", "n_unmeasured",
    "measured_enzymes", "unmeasured_enzymes", "shared_enzymes",
]


def get_isozyme_coverage(model: "EcModel") -> tuple[pd.DataFrame, list[str]]:
    """Report isozyme groups with partial proteomics coverage.

    The reactions in ``model.ec.rxns`` that share an identifier once the
    suffix ``_EXP_<n>`` is removed are the isozymes of one reaction of the
    conventional GEM. An entry is measured when at least one of its enzymes
    has a concentration above 0 in ``model.ec.concs``, and unmeasured when
    none of its enzymes has a concentration (all NaN). A group has partial
    coverage when it has both a measured and an unmeasured entry. An
    unmeasured entry next to a measured one can be a protein below the
    detection limit as well as a protein that is absent, so the report makes
    no statement on which. The kcat of an entry is not considered. A missing
    or short ``model.ec.concs`` counts as NaN for the enzymes it does not
    cover.

    Ported from GECKO MATLAB:
    src/geckomat/limit_proteins/getIsozymeCoverage.m.

    Parameters
    ----------
    model
        Full (not gecko-light) EcModel with concentrations in
        ``model.ec.concs`` (see ``fill_enz_concs``). Not modified.

    Returns
    -------
    coverage : pandas.DataFrame
        One row per group with partial coverage, sorted by reaction, with
        the columns:

        * ``rxn``: the reaction identifier without the ``_EXP_<n>`` suffix.
        * ``n_entries``: number of entries of the group in ``model.ec.rxns``.
        * ``n_measured``: number of measured entries.
        * ``n_unmeasured``: number of unmeasured entries.
        * ``measured_enzymes``: list of the enzymes of the measured entries
          that have a concentration above 0.
        * ``unmeasured_enzymes``: list of the enzymes of the unmeasured
          entries.
        * ``shared_enzymes``: list of the unmeasured enzymes that also
          catalyse an entry outside the unmeasured entries of groups with
          partial coverage. ``zero_unmeasured_isozymes`` leaves them
          unchanged.
    zeroable : list of str
        The enzymes that ``zero_unmeasured_isozymes`` acts on, in model
        order: the enzymes in ``unmeasured_enzymes`` that are not in
        ``shared_enzymes`` of any group.

    Raises
    ------
    NotImplementedError
        If the model is a gecko-light model.
    ValueError
        If ``model.ec.rxn_enz_mat`` is missing or empty.
    """
    if model.ec.gecko_light:
        raise NotImplementedError(
            "get_isozyme_coverage does not work on gecko-light models."
        )
    matrix = getattr(model.ec, "rxn_enz_mat", None)
    if matrix is None or 0 in matrix.shape:
        raise ValueError("The model has no ec.rxn_enz_mat. Run make_ec_model first.")

    enzymes = list(model.ec.enzymes)
    n_enz = len(enzymes)
    concs = np.full(n_enz, np.nan)
    given_concs = getattr(model.ec, "concs", None)
    given = np.asarray(
        given_concs if given_concs is not None else [], dtype=float,
    ).ravel()
    n_given = min(len(given), n_enz)
    concs[:n_given] = given[:n_given]
    matrix = (matrix != 0).astype(float).tocsr()

    positive = concs > 0
    missing = np.isnan(concs)
    entry_measured = matrix @ positive.astype(float) > 0
    entry_unmeasured = matrix @ (~missing).astype(float) == 0
    base_rxn = [re.sub(r"_EXP_\d+", "", rxn) for rxn in model.ec.rxns]
    groups, group_idx = np.unique(np.array(base_rxn, dtype=object), return_inverse=True)
    group_idx = np.asarray(group_idx).ravel()

    partial = np.zeros(len(groups), dtype=bool)
    for g in range(len(groups)):
        in_group = group_idx == g
        partial[g] = entry_measured[in_group].any() and entry_unmeasured[in_group].any()

    # An unmeasured entry of a group with partial coverage
    in_partial = entry_unmeasured & partial[group_idx]
    # An enzyme is zeroable when it is unmeasured and every entry it
    # catalyses is such an entry
    catalyses = matrix.T.tocsr()
    n_catalysed = np.asarray(catalyses.sum(axis=1)).ravel()
    n_in_partial = np.asarray(catalyses @ in_partial.astype(float)).ravel()
    zeroable_mask = missing & (n_catalysed > 0) & (n_in_partial == n_catalysed)
    zeroable = [enzymes[i] for i in np.where(zeroable_mask)[0]]

    rows = np.where(partial)[0]
    data: dict[str, list] = {c: [] for c in _COLUMNS}
    for g in rows:
        in_group = group_idx == g
        measured_rows = in_group & entry_measured
        unmeasured_rows = in_group & entry_unmeasured
        measured_mask = (np.asarray(matrix[measured_rows].sum(axis=0)).ravel() > 0) & positive
        unmeasured_mask = (np.asarray(matrix[unmeasured_rows].sum(axis=0)).ravel() > 0) & missing
        data["rxn"].append(str(groups[g]))
        data["n_entries"].append(int(in_group.sum()))
        data["n_measured"].append(int(measured_rows.sum()))
        data["n_unmeasured"].append(int(unmeasured_rows.sum()))
        data["measured_enzymes"].append([enzymes[i] for i in np.where(measured_mask)[0]])
        data["unmeasured_enzymes"].append([enzymes[i] for i in np.where(unmeasured_mask)[0]])
        data["shared_enzymes"].append(
            [enzymes[i] for i in np.where(unmeasured_mask & ~zeroable_mask)[0]]
        )
    coverage = pd.DataFrame(data, columns=_COLUMNS)
    for column in ("n_entries", "n_measured", "n_unmeasured"):
        coverage[column] = coverage[column].astype(int)
    return coverage, zeroable
