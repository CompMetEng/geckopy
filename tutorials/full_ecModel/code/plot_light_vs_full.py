"""Light versus full ecModel comparison and plot.

Python port of tutorials/full_ecModel/code/plotlightVSfull.m from
GECKO MATLAB. Builds a light and a full ecModel from the same
starting GEM with a deliberately simple kcat assignment (EC numbers
from the GEM, BRENDA via fuzzy matching, no tuning), so that the two
differ only in their formulation. Maximises growth on unlimited
glucose in both, maps the fluxes back to the starting GEM, and plots
the absolute fluxes of the light model against those of the full
model on log-log axes.

Functional equivalence with the MATLAB plot, not pixel parity.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cobra
import matplotlib.pyplot as plt
import numpy as np

from geckopy import (
    ModelAdapter,
    apply_kcat_constraints,
    apply_kcat_list,
    fill_eccodes_from_gem,
    fuzzy_kcat_matching,
    make_ec_model,
    map_rxns_to_conv,
    set_prot_pool_size,
)
from geckopy.databases import BrendaData, PhylDist, UniprotDB


_GLC_EXCHANGE = "r_1714"   # glucose uptake
_GROWTH_RXN = "r_2111"     # biomass-equation-coupled growth rxn
_ZERO_FLUX = 1e-8


def _build(
    model: cobra.Model,
    adapter: ModelAdapter,
    *,
    gecko_light: bool,
    uniprot_db: UniprotDB,
    brenda: BrendaData,
    phyl_dist: PhylDist,
):
    ec_model = make_ec_model(
        model.copy(), adapter, uniprot_db=uniprot_db, gecko_light=gecko_light,
    )
    fill_eccodes_from_gem(ec_model)
    apply_kcat_list(ec_model, fuzzy_kcat_matching(ec_model, brenda, phyl_dist))
    apply_kcat_constraints(ec_model)
    set_prot_pool_size(ec_model)
    return ec_model


def plot_light_vs_full(
    model: cobra.Model,
    adapter: ModelAdapter,
    *,
    uniprot_db: UniprotDB,
    brenda: BrendaData,
    phyl_dist: PhylDist,
    save_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare light and full ecModels at maximum growth rate.

    Parameters
    ----------
    model
        The starting GEM. It is copied, not modified.
    adapter
        The project's model adapter.
    uniprot_db, brenda, phyl_dist
        Pre-loaded UniProt, BRENDA and phylogenetic-distance data.
    save_path
        If supplied, the figure is saved to this path.

    Returns
    -------
    (flux_light, flux_full)
        Fluxes of the light and full ecModel at maximum growth rate,
        mapped onto ``model.reactions``, with ``|v| < 1e-8`` set to 0.
    """
    model = model.copy()
    model.reactions.get_by_id(_GLC_EXCHANGE).lower_bound = -1000
    model.objective = _GROWTH_RXN
    data = dict(uniprot_db=uniprot_db, brenda=brenda, phyl_dist=phyl_dist)

    timing: dict[str, list[float]] = {"build": [], "fba": [], "map": []}
    fluxes, growth = [], []
    for gecko_light in (False, True):
        t = time.perf_counter()
        ec_model = _build(model, adapter, gecko_light=gecko_light, **data)
        timing["build"].append(time.perf_counter() - t)

        t = time.perf_counter()
        sol = ec_model.optimize()
        timing["fba"].append(time.perf_counter() - t)

        t = time.perf_counter()
        mapped = map_rxns_to_conv(ec_model, model, sol.fluxes).mapped_flux
        timing["map"].append(time.perf_counter() - t)

        mapped = np.where(np.abs(mapped) < _ZERO_FLUX, 0.0, mapped)
        fluxes.append(mapped)
        growth.append(sol.objective_value)
    flux_full, flux_light = fluxes

    print("Duration light vs. full ecModel:")
    for step, label in (("build", "ecModel reconstruction"),
                        ("fba", "FBA"), ("map", "Mapping fluxes")):
        full_t, light_t = timing[step]
        print(f"  {label}: {100 * light_t / full_t:.0f}% "
              f"({light_t:.3f} vs {full_t:.3f} seconds)")
    print(f"Growth rate that is reached: {growth[0]:.4f} (full) vs "
          f"{growth[1]:.4f} (light)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.abs(flux_full), np.abs(flux_light), s=10)
    ax.set(xscale="log", yscale="log", xlim=(1e-8, 1e2), ylim=(1e-8, 1e2),
           xlabel="Full ecModel", ylabel="Light ecModel",
           title="Absolute fluxes (mmol/gDCWh)")
    ax.text(1e-7, 3, "Growth rate (/hour)")
    ax.text(1e-7, 0.5, f"full: {growth[0]:.4f}")
    ax.text(1e-7, 0.1, f"light: {growth[1]:.4f}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return flux_light, flux_full
