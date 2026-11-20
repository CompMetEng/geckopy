"""A fluxData.tsv file feeds apply_flux_data_constraints."""
from pathlib import Path

import numpy as np
import pytest

from raven_toolbox.experiments import load_flux_data


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_loaded_data_can_feed_apply_flux_data_constraints(tmp_path):
    """End-to-end: load a fluxData.tsv, constrain a model with it."""
    import cobra
    from scipy import sparse

    from geckopy import EcModel, ModelAdapter
    from geckopy.ec_model.ec_data import EcData
    from geckopy.limit_proteins import apply_flux_data_constraints

    (tmp_path / "model_adapter.toml").write_text(
        'conv_gem = "dummy.xml"\n'
        'org_name = "test"\n'
        'bio_rxn = "biomass"\n'
        'c_source = "EX_glc"\n'
    )
    adapter = ModelAdapter.from_folder(tmp_path)

    # Build a tiny model with one exchange and biomass.
    model = EcModel("test", adapter=adapter)
    glc = cobra.Metabolite("glc_e", compartment="e")
    bio = cobra.Metabolite("bio", compartment="c")
    model.add_metabolites([glc, bio])
    EX_glc = cobra.Reaction("EX_glc")
    EX_glc.add_metabolites({glc: -1.0})
    EX_glc.lower_bound = -1000.0
    EX_glc.upper_bound = 0.0
    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio: 1.0})
    BIO.lower_bound = 0.0
    BIO.upper_bound = 1000.0
    model.add_reactions([EX_glc, BIO])
    model.ec = EcData(
        rxns=[], kcat=np.empty(0), source=[], notes=[], eccodes=[],
        rxn_enz_mat=sparse.csr_matrix((0, 0)),
    )

    # Write a fluxData.tsv with a different exchange (so c_source gets zeroed).
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glucose (EX_glc)"],
        ["c1", "0.5", "0.4", "-5.0"],
    ])
    fd = load_flux_data(p)
    apply_flux_data_constraints(model, fd)
    assert model.reactions.get_by_id("EX_glc").lower_bound == -5.0
    assert model.reactions.get_by_id("biomass").upper_bound == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Malformed rows and headers
# --------------------------------------------------------------------------- #
