"""The geckopy side of the experiment data: the legacy weight column and the SMILES wrapper."""
import pytest

from geckopy.databases import smiles
from geckopy.kcat_tuning.evotune.tuning import _reject_legacy_weight_column

_HEADER = "Condition\tPtot\tgrRate\tglucose (r_1714)\t{weight}\tsource\n"
_ROW = "glucose\tNaN\t0.36\t-13.33\t1\tx\n"


def _write(folder, name, weight_column):
    (folder / name).write_text(_HEADER.format(weight=weight_column) + _ROW)


@pytest.mark.parametrize("name", ["fluxData.tsv", "maxGrowth.tsv"])
def test_the_legacy_weight_column_is_rejected_with_the_new_name(tmp_path, name):
    _write(tmp_path, name, "evotuneRMSEweight")
    with pytest.raises(ValueError, match="evotuneRMSEweight.*'weight'"):
        _reject_legacy_weight_column(tmp_path)


def test_the_weight_column_and_missing_files_are_accepted(tmp_path):
    _reject_legacy_weight_column(tmp_path)
    _write(tmp_path, "fluxData.tsv", "weight")
    _reject_legacy_weight_column(tmp_path)


def test_find_met_smiles_skips_the_enzyme_metabolites_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        smiles, "_find_met_smiles", lambda model, **kwargs: calls.append(kwargs)
    )

    smiles.find_met_smiles("model", cache_path=tmp_path / "db.tsv")
    smiles.find_met_smiles("model", cache_path=tmp_path / "db.tsv", skip_prefixes=("x_",))

    assert calls[0]["skip_prefixes"] == ("prot_",)
    assert calls[1]["skip_prefixes"] == ("x_",)
    assert calls[0]["cache_path"] == tmp_path / "db.tsv"
