"""Tests for the behaviour GECKO's kcatContractTests check as well."""
import shutil
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.databases import UniprotDB, load_dlkcat_ignore_lists
from geckopy.gather_kcats import (
    assign_standard_kcat,
    kcats_from_ec_tree,
    read_dlkcat_output,
    read_kcat_source,
    write_dlkcat_input,
)
from geckopy.gather_kcats.ec_tree import ECTree
from geckopy.get_enzyme_data.ec_from_gem import fill_eccodes_from_gem

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


def _project(tmp_path: Path, extra_reactions: bool = False, light: bool = False):
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    adapter = ModelAdapter.from_folder(project)
    conv = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    if extra_reactions:
        m1 = conv.metabolites.get_by_id("m1c")
        m2 = conv.metabolites.get_by_id("m2c")
        for rxn_id, name, stoich in (("R6", "R6", {m1: -1.0, m2: 1.0}),
                                     ("R7", "R7", {m1: -1.0, m2: -1.0}),
                                     ("R8", "Biomass Pseudoreaction", {m1: -1.0, m2: 1.0})):
            rxn = cobra.Reaction(rxn_id, name=name)
            rxn.lower_bound = 0.0
            rxn.upper_bound = 1000.0
            rxn.add_metabolites(stoich)
            conv.add_reactions([rxn])
    model = make_ec_model(conv, adapter, gecko_light=light)
    fill_eccodes_from_gem(model)
    return model, adapter, project


def _uniprot() -> UniprotDB:
    mws = [100.0, 200.0, 300.0]
    return UniprotDB(ids=["P000", "P001", "P002"], genes=["g0", "g1", "g2"], eccodes=["1.1.1.1"] * 3,
                     mw=np.array(mws), sequences=[""] * 3)


def test_read_kcat_source_quoted_rows(tmp_path):
    """A row of empty quoted fields is blank and a quoted comment is a comment; a
    field that opens a quote and does not close it is an error."""
    file = tmp_path / "odd.tsv"
    head = ["# source: odd", "rxn\tenzymes\tkcat\tsource", "R3\tP4\t25\todd"]
    file.write_text("\n".join(head + ['"# a quoted comment"', '""\t""\t""\t""', "R5\tP5\t30\todd"]) + "\n",
                    encoding="utf-8")
    records, _ = read_kcat_source(file)
    assert list(records["kcat"]) == [25.0, 30.0]
    assert list(records["line"]) == [3, 6]
    file.write_text("\n".join(head + ['"# a comment that was quoted']) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 4"):
        read_kcat_source(file)
    file.write_text("\n".join(head[:2] + ['"R3\tP4\t25\todd']) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no closing one"):
        read_kcat_source(file)
