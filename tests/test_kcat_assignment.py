"""Tests for collect_kcats, assign_kcats and update_kcat_sources, on ecTestGEM.

GECKO's kcatAssignmentTests (tc0079-tc0082) test the same.
"""
import shutil
from pathlib import Path

import cobra
import numpy as np
import pydantic
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.databases import load_brenda_data, load_phyl_dist, load_uniprot_tsv
from geckopy.ec_model.pipeline import apply_custom_kcats, apply_kcat_constraints, fill_kcats_from_isozymes
from geckopy.gather_kcats import (
    apply_kcat_list,
    assign_kcats,
    assign_standard_kcat,
    collect_kcats,
    ec_rxn_keys,
    fuzzy_kcat_matching,
    kcats_from_brenda,
    kcats_from_dlkcat,
    merge_kcats,
    read_dlkcat_output,
    read_kcat_source,
    update_kcat_sources,
)

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"

DLKCAT = (
    "rxns\tgenes\tsubstrates\tsmiles\tsequence\tkcat\n"
    "R2_EXP_1\tG1\tm1\tC\tMRAL\t12.5\n"
    "R2_EXP_1\tG2\tm1\tC\tMNTD\t6.5\n"
    "R3\tG4\tm1\tC\tMDFM\t3.25\n"
)


def _project(tmp_path: Path, toml_extra: str = ""):
    """ecTestGEM in tmp_path, with its BRENDA files as the project's BRENDA
    data and DLKcat output for the test model."""
    project = tmp_path / "ecTestGEM"
    shutil.copytree(EXAMPLE_DIR, project)
    brenda = project / "data" / "brenda"
    brenda.mkdir()
    for name in ("kcat.tsv", "sa.tsv", "mw.tsv"):
        shutil.copy(project / "data" / name, brenda / name)
    (project / "data" / "DLKcat.tsv").write_text(DLKCAT, encoding="utf-8")
    if toml_extra:
        toml = project / "model_adapter.toml"
        toml.write_text(toml.read_text(encoding="utf-8") + "\n" + toml_extra, encoding="utf-8")
    adapter = ModelAdapter.from_folder(project)
    model = make_ec_model(cobra.io.read_sbml_model(str(adapter.params.conv_gem)), adapter)
    return model, adapter, project


def test_assign_kcats_reproduces_chain(tmp_path):
    """gecko_classic gives the same model as the steps it replaces."""
    base, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "gecko_classic"\n')
    data = project / "data"
    brenda = load_brenda_data(data / "brenda")
    phyl = load_phyl_dist(data / "PhylDist.mat")
    uniprot = load_uniprot_tsv(data / "uniprot.tsv")

    a = base.copy()
    fuzzy = fuzzy_kcat_matching(a, brenda, phyl, adapter=adapter)
    dlkcat = read_dlkcat_output(a, data / "DLKcat.tsv")
    merged = merge_kcats(fuzzy, dlkcat, source_priority=["database_top", "dlkcat", "database_bottom"],
                         top_origin_limit=6)
    apply_kcat_list(a, merged)
    apply_custom_kcats(a, path=data / "customKcats.tsv")
    fill_kcats_from_isozymes(a)
    assign_standard_kcat(a, uniprot)
    apply_kcat_constraints(a)

    b = base.copy()
    kcats_from_brenda(b, brenda=brenda, phyl_dist=phyl, adapter=adapter)
    kcats_from_dlkcat(b, adapter=adapter)
    decisions, report = assign_kcats(b, adapter=adapter, uniprot_db=uniprot)
    assert list(b.ec.rxns) == list(a.ec.rxns)
    np.testing.assert_array_equal(b.ec.kcat, a.ec.kcat)
    assert list(b.ec.source) == list(a.ec.source)
    stoich = lambda m: {r.id: {x.id: c for x, c in r.metabolites.items()} for r in m.reactions}  # noqa: E731
    assert stoich(b) == stoich(a)
    assert report["preset"] == "gecko_classic"
    assert list(decisions["ec_rxn"]) == list(b.ec.rxns)
    assert (project / "output" / f"{base.id}_kcatDecisions.tsv").is_file()
    assert {"custom", "1"} <= set(decisions["stage"])


def test_gecko_median_preset(tmp_path):
    """gecko_median: the BRENDA median, and BRENDA matches beyond origin 4
    rank below DLKcat."""
    base, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "gecko_median"\n')
    data = project / "data"
    uniprot = load_uniprot_tsv(data / "uniprot.tsv")
    a = base.copy()
    fuzzy = kcats_from_brenda(a, aggregation="median", adapter=adapter)
    dlkcat = kcats_from_dlkcat(a, adapter=adapter)
    merged = merge_kcats(fuzzy, dlkcat, source_priority=["database_top", "dlkcat", "database_bottom"],
                         top_origin_limit=4)
    apply_kcat_list(a, merged)
    apply_custom_kcats(a, path=data / "customKcats.tsv")
    fill_kcats_from_isozymes(a)
    assign_standard_kcat(a, uniprot)
    apply_kcat_constraints(a)
    b = base.copy()
    _, report = assign_kcats(b, adapter=adapter, uniprot_db=uniprot, write_report=False)
    assert report["preset"] == "gecko_median"
    np.testing.assert_array_equal(b.ec.kcat, a.ec.kcat)
    assert list(b.ec.source) == list(a.ec.source)


def test_collect_kcats_leaves_model(tmp_path):
    """collect_kcats decides on a copy; a missing source is reported;
    custom.tsv decides first."""
    base, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "gecko_median"\n')
    kcats_from_brenda(base, adapter=adapter)
    decisions, report, decided = collect_kcats(base, adapter=adapter, write_report=False)
    assert not np.any(base.ec.kcat)
    assert report["missing"] == ["dlkcat"]
    assert np.any(decided.ec.kcat > 0)
    assert not (project / "output").exists()

    rxns, enzymes = ec_rxn_keys(base)
    i = base.ec.rxns.index("R3")
    (project / "data" / "kcats" / "custom.tsv").write_text(
        "# source: custom\n# kind: curated\nrxn\tenzymes\tkcat\tsource\n"
        f"{rxns[i]}\t{enzymes[i]}\t777\tcustom\n", encoding="utf-8")
    decisions, _, _ = collect_kcats(base, adapter=adapter, write_report=False)
    row = decisions.iloc[i]
    assert (row["kcat"], row["source"], row["stage"]) == (777, "custom", "1")


def test_stages_from_toml(tmp_path):
    """Stages from model_adapter.toml, and their errors."""
    extra = ('[kcat_assignment]\nisozymes = false\nstandard = false\n'
             '[[kcat_assignment.stage]]\nsources = ["dlkcat"]\ncriteria = "min"\n'
             '[kcat_sources.brenda]\naggregation = "median"\n')
    base, adapter, project = _project(tmp_path, extra)
    kcats_from_dlkcat(base, adapter=adapter)
    decisions, report, _ = collect_kcats(base, adapter=adapter, write_report=False)
    assert report["preset"] == ""
    assert set(decisions.loc[decisions["kcat"] > 0, "source"]) == {"dlkcat"}
    assert decisions.loc[decisions["ec_rxn"] == "R2_EXP_1", "kcat"].item() == 6.5

    params = ModelAdapter.from_folder(project).params
    assert params.kcat_assignment.standard is False
    assert params.kcat_sources["brenda"]["aggregation"] == "median"
    toml = project / "model_adapter.toml"
    text = toml.read_text(encoding="utf-8")
    toml.write_text(text.replace("isozymes = false", 'preset = "gecko_median"'), encoding="utf-8")
    with pytest.raises(pydantic.ValidationError, match="both a preset and stages"):
        ModelAdapter.from_folder(project)
    toml.write_text(text.replace("isozymes = false", 'preset = "gecko_fast"')
                    .split("[[kcat_assignment.stage]]")[0], encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        ModelAdapter.from_folder(project)


def test_update_kcat_sources(tmp_path):
    """Missing and outdated sources are gathered; current ones are not."""
    base, adapter, project = _project(tmp_path, '[kcat_assignment]\npreset = "gecko_median"\n')
    report = update_kcat_sources(base, dry_run=True, adapter=adapter)
    assert list(report["source"]) == ["custom", "brenda", "dlkcat"]
    assert list(report["status"]) == ["skipped", "missing", "missing"]
    assert not (project / "data" / "kcats").exists()
    report = update_kcat_sources(base, adapter=adapter)
    assert list(report["status"]) == ["skipped", "gathered", "gathered"]
    _, meta = read_kcat_source(project / "data" / "kcats" / "brenda.tsv")
    assert meta["parameters"] == '{"aggregation":"median","force_wildcard_level":0}'
    report = update_kcat_sources(base, adapter=adapter)
    assert list(report["status"]) == ["skipped", "current", "current"]
    # gecko_classic asks for the BRENDA maximum: brenda.tsv is outdated.
    adapter.params.kcat_assignment.preset = "gecko_classic"
    report = update_kcat_sources(base, dry_run=True, adapter=adapter)
    assert report["status"].iloc[1] == "outdated"
