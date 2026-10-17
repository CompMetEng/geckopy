"""Tests for the combine stage and the OKP presets.

GECKO's combineTests test the same numbers.
"""
import math
import shutil
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.gather_kcats import collect_kcats, save_kcat_source
from geckopy.gather_kcats.combine_kcats import combine_kcats, evidence_bin, load_combine_params
from geckopy.gather_kcats.kcat_assignment import kcat_assignment_config, okp_file, update_kcat_sources
from geckopy.get_enzyme_data.ec_from_gem import fill_eccodes_from_gem

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"

PARAMS = """# Combination parameters for tests
# measurement_sd: 0.5
# min_variance: 0.01
# interval_scale ec_tree+okp: 2
# interval_scale okp: 1
source\tbin\tbias\tvariance\tn
ec_tree\tec4:3-5\t0.1\t1\t50
ec_tree\tec3\t0\t4\t50
okp_catapro\t<40\t0.2\t2\t50
okp_catapro\t>80\t0\t0.5\t50
okp_unikp\t40-60\t-0.3\t1\t50
"""


def _params(tmp_path):
    f = tmp_path / "kcatCombineParams.tsv"
    f.write_text(PARAMS, encoding="utf-8")
    return load_combine_params(f)


def test_evidence_bins():
    assert evidence_bin("ec_tree", "n=4; depth=4") == "ec4:3-5"
    assert evidence_bin("ec_tree", "n=150; depth=4") == "ec4:>100"
    assert evidence_bin("ec_tree", "n=0; depth=3") == "ec3"
    assert evidence_bin("okp_catapro", "similarity=40") == "<40"
    assert evidence_bin("okp_catapro", "similarity=40.5") == "40-60"
    assert evidence_bin("okp_catapro", "similarity=95") == ">80"
    assert evidence_bin("okp_catapro", "") == ""
    assert evidence_bin("homologs", "identity=0.85") == "80"


def test_combine_worked_example(tmp_path):
    """Reaction R1: EC tree 10 /s (ec4:3-5, bias 0.1, variance 1) and CataPro
    100 /s (similarity 30: <40, bias 0.2, variance 2). Corrected log10 values
    0.9 and 1.8, weights 1 and 0.5: mean (0.9 + 0.9) / 1.5 = 1.2; sd
    sqrt(1/1.5 + 0.25) * 2 (the ec_tree+okp scale)."""
    p = _params(tmp_path)
    lists = {
        "ec_tree": pd.DataFrame({"rxn_id": ["R1", "R2"], "kcat": [10.0, 1.0], "detail": ["n=4; depth=4", "n=0; depth=3"]}),
        # Two rows for R1 (substrates): their median, with the bin of best evidence.
        "okp_catapro": pd.DataFrame({"rxn_id": ["R1", "R1"], "kcat": [100.0, 100.0],
                                     "detail": ["similarity=30", "similarity=20"]}),
    }
    c = combine_kcats(lists, p).set_index("rxn_id")
    assert c.loc["R1", "kcat"] == pytest.approx(10 ** 1.2)
    assert c.loc["R1", "sd"] == pytest.approx(math.sqrt(1 / 1.5 + 0.25) * 2)
    assert c.loc["R1", "n"] == 2 and c.loc["R1", "spread_log10"] == pytest.approx(1.0)
    # R2 has the EC tree only (ec3: bias 0): its value, with no scale for ec_tree alone.
    assert c.loc["R2", "kcat"] == pytest.approx(1.0)
    assert c.loc["R2", "sd"] == pytest.approx(math.sqrt(4 + 0.25))
    # A value without a known bin takes the source's largest variance (CataPro <40: 2, bias 0.2).
    lists["okp_catapro"] = pd.DataFrame({"rxn_id": ["R3"], "kcat": [10.0], "detail": [""]})
    c = combine_kcats(lists, p).set_index("rxn_id")
    assert c.loc["R3", "kcat"] == pytest.approx(10 ** 0.8)
    with pytest.raises(ValueError, match="okp_eitlem"):
        combine_kcats({"okp_eitlem": lists["okp_catapro"]}, p)


def _project(tmp_path: Path, preset: str):
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    (brenda / "kcatCombineParams.tsv").write_text(PARAMS, encoding="utf-8")
    (project / "data" / "customKcats.tsv").unlink()
    toml = project / "model_adapter.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + f'\n[kcat_assignment]\npreset = "{preset}"\n'
                    '[kcat_sources.okp]\nmethods = ["CataPro", "UniKP"]\n', encoding="utf-8")
    adapter = ModelAdapter.from_folder(project)
    model = make_ec_model(cobra.io.read_sbml_model(str(adapter.params.conv_gem)), adapter)
    fill_eccodes_from_gem(model)
    return model, adapter, project


def _okp_list(model, rxns, kcat, similarity, method):
    return pd.DataFrame({"rxn_id": rxns, "source": f"OKP-{method}", "kcat": kcat,
                         "substrates": [[] for _ in rxns], "detail": [f"similarity={similarity}"] * len(rxns)})


def test_okp_presets(tmp_path):
    model, adapter, project = _project(tmp_path, "okp")
    config = kcat_assignment_config(adapter)
    assert [st.action for st in config.stages] == ["select", "combine"]
    assert okp_file("KinForm-H") == "okp_kinform_h"
    report = update_kcat_sources(model, dry_run=True, adapter=adapter)
    assert list(report["source"]) == ["custom", "okp_catapro", "okp_unikp"]

    rxns = [r for r in model.ec.rxns if r.startswith("R")][:3]
    save_kcat_source(model, _okp_list(model, rxns, 100.0, 30, "CataPro"), "okp_catapro", adapter=adapter)
    save_kcat_source(model, _okp_list(model, rxns[:1], 10.0, 50, "UniKP"), "okp_unikp", adapter=adapter)
    decisions, rep, decided = collect_kcats(model, adapter=adapter, write_report=False)
    d = decisions.set_index("ec_rxn")
    # rxns[0]: CataPro 100 (bias 0.2, variance 2) and UniKP 10 (bias -0.3, variance 1):
    # (1.8 * 0.5 + 1.3 * 1) / 1.5 = 1.4667; the okp scale is 1.
    assert d.loc[rxns[0], "kcat"] == pytest.approx(10 ** ((1.8 * 0.5 + 1.3) / 1.5))
    assert d.loc[rxns[0], "source"] == f"combined (sd={math.sqrt(1 / 1.5 + 0.25):.2f}, n=2)"
    assert d.loc[rxns[0], "stage"] == "2" and d.loc[rxns[0], "candidates"] == 2
    assert d.loc[rxns[1], "kcat"] == pytest.approx(10 ** 1.8)
    assert np.isfinite(d.loc[rxns[0], "sd_log10"])

    # ec_tree_okp: the EC tree and OKP combined, where both have a value.
    model, adapter, project = _project(tmp_path / "b", "ec_tree_okp")
    save_kcat_source(model, _okp_list(model, rxns[:1], 100.0, 30, "CataPro"), "okp_catapro", adapter=adapter)
    i = model.ec.rxns.index(rxns[0])
    ec_list = pd.DataFrame({"rxn_id": [rxns[0]], "source": "ec_tree", "kcat": [10.0], "substrates": [[]],
                            "eccode": [model.ec.eccodes[i]], "detail": ["n=4; depth=4"]})
    save_kcat_source(model, ec_list, "ec_tree", adapter=adapter)
    decisions, _, _ = collect_kcats(model, adapter=adapter, write_report=False)
    d = decisions.set_index("ec_rxn")
    assert d.loc[rxns[0], "kcat"] == pytest.approx(10 ** 1.2)
    assert d.loc[rxns[0], "source"].startswith("combined (") and d.loc[rxns[0], "stage"] == "2"


def test_unknown_okp_method(tmp_path):
    model, adapter, project = _project(tmp_path, "okp")
    adapter.params.kcat_sources["okp"]["methods"] = ["CataPro", "NoSuchMethod"]
    with pytest.raises(ValueError, match="NoSuchMethod"):
        update_kcat_sources(model, dry_run=True, adapter=adapter)
