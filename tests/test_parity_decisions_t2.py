"""make_ec_model refuses what GECKO's makeEcModel refuses and treats reversibility and
exchange reactions as it does; set_kcat_for_reactions follows setKcatForReactions."""
from pathlib import Path

import cobra
import pytest

from geckopy import ModelAdapter, make_ec_model
from geckopy.ec_model.pipeline import set_kcat_for_reactions

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "ecTestGEM"


def _inputs():
    adapter = ModelAdapter.from_folder(EXAMPLE_DIR)
    model = cobra.io.read_sbml_model(str(adapter.params.conv_gem))
    return model, adapter


def _build(edit=None, *, light=False, adapter_edit=None):
    model, adapter = _inputs()
    if edit is not None:
        edit(model)
    if adapter_edit is not None:
        adapter_edit(adapter)
    return make_ec_model(model, adapter, gecko_light=light)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

def test_the_unedited_model_builds():
    assert _build().ec.n_rxns > 0
    assert _build(light=True).ec.n_rxns > 0


def _rename_rxn(old, new):
    return lambda model: setattr(model.reactions.get_by_id(old), "id", new)


@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("old,new", [
    ("R1", "R1_REV"),
    ("R1", "R1_EXP_1"),
    ("R5", "usage_prot_P5"),
    ("R5", "prot_pool_R5"),
])
def test_reaction_id_conflicts_are_refused(old, new, light):
    with pytest.raises(ValueError, match="model.reactions"):
        _build(_rename_rxn(old, new), light=light)


def test_a_reaction_id_with_the_suffix_text_inside_is_accepted():
    ec_model = _build(_rename_rxn("R1", "R_REVX"))
    assert "R_REVX" in ec_model.reactions


@pytest.mark.parametrize("light", [False, True])
def test_metabolite_id_starting_with_prot_is_refused(light):
    def edit(model):
        model.metabolites.get_by_id("m2c").id = "prot_m2"
    with pytest.raises(ValueError, match="prot_"):
        _build(edit, light=light)


def test_a_compartment_id_that_differs_from_its_name_is_refused():
    def edit(model):
        model.compartments = {"c": "cytosol", "e": "extra"}
    with pytest.raises(ValueError, match="compartment names"):
        _build(edit)


def test_an_unknown_enzyme_compartment_is_refused():
    def adapter_edit(adapter):
        adapter.params.enzyme_comp = "zz"
    with pytest.raises(ValueError, match="compartment names"):
        _build(adapter_edit=adapter_edit)


@pytest.mark.parametrize("light", [False, True])
def test_a_model_without_genes_is_refused(light):
    def edit(model):
        for rxn in model.reactions:
            rxn.gene_reaction_rule = ""
        cobra.manipulation.delete.remove_genes(
            model, list(model.genes), remove_reactions=False
        )
    with pytest.raises(ValueError, match="no genes"):
        _build(edit, light=light)


# --------------------------------------------------------------------------- #
# Reversibility and exchange reactions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("light", [False, True])
def test_a_reaction_with_both_bounds_negative_is_not_split(light):
    def edit(model):
        model.reactions.get_by_id("R3").bounds = (-10.0, -1.0)
    ec_model = _build(edit, light=light)
    assert "R3_REV" not in ec_model.reactions
    assert ec_model.reactions.get_by_id("R3").bounds == (-10.0, -1.0)


def _add_sink(model):
    sink = cobra.Reaction("SINK2", lower_bound=-1000.0, upper_bound=1000.0)
    model.add_reactions([sink])
    sink.add_metabolites({
        model.metabolites.get_by_id("m1c"): -1.0,
        model.metabolites.get_by_id("m2c"): -1.0,
    })


@pytest.mark.parametrize("light", [False, True])
def test_a_multi_metabolite_sink_is_an_exchange_reaction_and_stays_reversible(light):
    ec_model = _build(_add_sink, light=light)
    assert "SINK2_REV" not in ec_model.reactions
    assert ec_model.reactions.get_by_id("SINK2").bounds == (-1000.0, 1000.0)


def test_a_reversible_internal_reaction_is_still_split():
    ec_model = _build()
    assert "R2_REV_EXP_1" in ec_model.reactions


# --------------------------------------------------------------------------- #
# set_kcat_for_reactions
# --------------------------------------------------------------------------- #

def _ec_model(light=False):
    return _build(light=light)


def _kcat(ec_model, rxn_id):
    return float(ec_model.ec.kcat[ec_model.ec.rxns.index(rxn_id)])


def test_set_kcat_does_not_apply_the_constraints_by_default():
    ec_model = _ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0)
    coefficients = {m.id: c for m, c in ec_model.reactions.get_by_id("R3").metabolites.items()}
    assert "prot_P4" not in coefficients
    assert _kcat(ec_model, "R3") == 10.0


def test_set_kcat_applies_the_constraints_when_asked():
    ec_model = _ec_model()
    set_kcat_for_reactions(ec_model, ["R3"], 10.0, apply=True)
    coefficients = {m.id: c for m, c in ec_model.reactions.get_by_id("R3").metabolites.items()}
    assert coefficients["prot_P4"] == pytest.approx(-(40000.0 / (10.0 * 3600.0)))


def test_set_kcat_matches_ids_without_regard_to_letter_case():
    ec_model = _ec_model()
    assert set_kcat_for_reactions(ec_model, ["r3"], 7.0) == ["R3"]
    assert _kcat(ec_model, "R3") == 7.0
    assert set_kcat_for_reactions(ec_model, ["r2_EXP_1"], 8.0) == ["R2_EXP_1"]
    assert _kcat(ec_model, "R2_EXP_1") == 8.0


def test_set_kcat_ignores_an_unknown_id():
    ec_model = _ec_model()
    assert set_kcat_for_reactions(ec_model, ["nonsense"], 1.0) == []
    assert set_kcat_for_reactions(ec_model, ["R3", "nonsense"], 1.0) == ["R3"]
    assert _kcat(ec_model, "R3") == 1.0


def test_set_kcat_on_a_light_model_does_not_expand_a_bare_cobra_id():
    ec_model = _ec_model(light=True)
    assert set_kcat_for_reactions(ec_model, ["R2"], 5.0) == []
    assert _kcat(ec_model, "001_R2") == 0.0
    assert set_kcat_for_reactions(ec_model, ["001_R2", "002_R2"], 5.0) == ["001_R2", "002_R2"]
    assert _kcat(ec_model, "002_R2") == 5.0
