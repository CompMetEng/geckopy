"""Tests for get_conc_control_coeffs."""
import cobra
import numpy as np
import pytest
from scipy import sparse

from geckopy.ec_model import EcModel
from geckopy.ec_model.ec_data import EcData
from geckopy.limit_proteins import get_conc_control_coeffs


# --------------------------------------------------------------------------- #
# Tiny enzyme-constrained model fixture
# --------------------------------------------------------------------------- #

def _build_toy_ec_model(
    *,
    enzyme_ub: float = 5.0,
    enzyme_coeff_in_R: float = -1.0,
) -> EcModel:
    """Build a 1-substrate-1-enzyme model with a measurable growth rate.

    Topology:
        EX_A:               A_e <->                  (substrate exchange)
        TR_A:               A_e -> A_c               (transport)
        R_AB:    A_c + (-coeff)*prot_E -> B_c        (enzyme-constrained)
        SK_B:               B_c ->                   (sink, biomass-like)
        prot_pool_exchange: prot_pool ->             (pool source)
        usage_prot_E:       prot_pool -> prot_E      (usage; ub = enzyme_ub)

    The objective is SK_B. Maximum SK_B flux equals max usage flux,
    which equals enzyme_ub when enzyme_coeff_in_R = -1 (one unit of
    enzyme per unit of R flux).
    """
    model = EcModel("toy")

    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    model.add_metabolites([A_e, A_c, B_c, pool, prot_E])

    EX_A = cobra.Reaction("EX_A")
    EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = -1000.0
    EX_A.upper_bound = 0.0

    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0
    TR_A.upper_bound = 1000.0

    R_AB = cobra.Reaction("R_AB")
    R_AB.add_metabolites({A_c: -1.0, B_c: 1.0, prot_E: enzyme_coeff_in_R})
    R_AB.lower_bound = 0.0
    R_AB.upper_bound = 1000.0

    SK_B = cobra.Reaction("SK_B")
    SK_B.add_metabolites({B_c: -1.0})
    SK_B.lower_bound = 0.0
    SK_B.upper_bound = 1000.0

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0
    pool_ex.upper_bound = 1000.0

    usage = cobra.Reaction("usage_prot_E")
    usage.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage.lower_bound = 0.0
    usage.upper_bound = enzyme_ub

    model.add_reactions([EX_A, TR_A, R_AB, SK_B, pool_ex, usage])
    model.objective = "SK_B"

    model.ec = EcData(
        rxns=["R_AB"],
        kcat=np.array([1.0]),
        source=[""],
        notes=[""],
        eccodes=[""],
        genes=["g_E"],
        enzymes=["E"],
        mw=np.array([100.0]),
        sequence=[""],
        concs=np.array([float(enzyme_ub)]),
        rxn_enz_mat=sparse.csr_matrix([[1.0]]),
    )
    return model


# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_empty_proteins_list_returns_empty():
    model = _build_toy_ec_model()
    enz, coeffs = get_conc_control_coeffs(model, proteins=[])
    assert enz.size == 0
    assert coeffs.size == 0


def test_default_proteins_uses_model_ec_enzymes():
    model = _build_toy_ec_model()
    enz, coeffs = get_conc_control_coeffs(model)  # default proteins
    assert len(enz) == 1
    assert len(coeffs) == 1


def test_protein_not_in_model_silently_skipped():
    model = _build_toy_ec_model()
    enz, coeffs = get_conc_control_coeffs(
        model, proteins=["E", "ghost"],
    )
    assert enz[0]
    assert not enz[1]
    assert coeffs[1] == 0.0


# --------------------------------------------------------------------------- #
# Sensitivity analysis: limiting enzyme
# --------------------------------------------------------------------------- #

def test_limiting_enzyme_marked_and_has_positive_coeff():
    """The single enzyme E directly limits R_AB; relaxing it must
    yield a positive control coefficient."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    enz, coeffs = get_conc_control_coeffs(model)
    assert enz[0]
    assert coeffs[0] > 0


def test_control_coeff_matches_growth_per_unit_increase():
    """With enzyme_coeff_in_R = -1, growth = enzyme ub. Doubling ub
    from 5 to 10 increases growth by 5; coefficient = 5/(10-5) = 1."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    _, coeffs = get_conc_control_coeffs(
        model, fold_change=2.0,
    )
    assert coeffs[0] == pytest.approx(1.0, rel=1e-6)


def test_control_coeff_scales_with_enzyme_coefficient():
    """If R_AB consumes 2 units of enzyme per unit flux, doubling the
    enzyme ub from 5 to 10 increases growth from 2.5 to 5 (delta=2.5),
    and the coeff is 2.5/(10-5) = 0.5."""
    model = _build_toy_ec_model(enzyme_ub=5.0, enzyme_coeff_in_R=-2.0)
    _, coeffs = get_conc_control_coeffs(model, fold_change=2.0)
    assert coeffs[0] == pytest.approx(0.5, rel=1e-6)


# --------------------------------------------------------------------------- #
# limit parameter
# --------------------------------------------------------------------------- #

def test_default_limit_zero_includes_active_enzyme():
    """Default limit=0 -> any protein with non-zero usage qualifies."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    enz, _ = get_conc_control_coeffs(model, limit=0.0)
    assert enz[0]


def test_high_limit_excludes_under_used_enzyme():
    """Set ub very high so usage/ub is small; with limit=0.5, the
    enzyme should be skipped."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    # Cap upstream so the enzyme isn't used at full capacity:
    # restrict EX_A to deliver only 1 unit/time, so usage flux = 1.
    model.reactions.get_by_id("EX_A").lower_bound = -1.0
    enz, coeffs = get_conc_control_coeffs(model, limit=0.5)
    assert not enz[0]
    assert coeffs[0] == 0.0


# --------------------------------------------------------------------------- #
# fold_change parameter
# --------------------------------------------------------------------------- #

def test_fold_change_affects_probe_size():
    """Coefficient = delta_growth / delta_ub. With enzyme_coeff_in_R=-1,
    delta_growth = delta_ub, so the coefficient is 1.0 regardless of
    fold_change. Verify both default and a different value."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    _, c2 = get_conc_control_coeffs(model, fold_change=2.0)
    _, c5 = get_conc_control_coeffs(model, fold_change=5.0)
    assert c2[0] == pytest.approx(1.0, rel=1e-6)
    assert c5[0] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Reversibility: model state restored after each probe
# --------------------------------------------------------------------------- #

def test_model_bounds_unchanged_after_call():
    model = _build_toy_ec_model(enzyme_ub=5.0)
    original_ub = model.reactions.get_by_id("usage_prot_E").upper_bound
    get_conc_control_coeffs(model)
    assert model.reactions.get_by_id("usage_prot_E").upper_bound == original_ub


# --------------------------------------------------------------------------- #
# Subset of proteins
# --------------------------------------------------------------------------- #

def test_subset_proteins_only_those_evaluated():
    model = _build_toy_ec_model(enzyme_ub=5.0)
    enz, coeffs = get_conc_control_coeffs(model, proteins=["E"])
    assert len(enz) == 1
    assert enz[0]
    assert coeffs[0] > 0


# --------------------------------------------------------------------------- #
# Single-solve alternative (validating the review §3.4 suggestion)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "ub, coeff_in_R, expected",
    [(5.0, -1.0, 1.0), (3.0, -1.0, 1.0), (6.0, -2.0, 0.5)],
)
def test_shadow_price_matches_finite_difference_coeff(ub, coeff_in_R, expected):
    """The per-protein finite-difference loop can be replaced by a single
    solve: the prot_<id> metabolite shadow price (|dual|) equals the
    finite-difference control coefficient.

    Note: the *reaction* ``reduced_cost`` of the usage reaction does NOT —
    it is scaled by the number of coupled pseudometabolites (prot_pool +
    prot_<id>), so it reads 2x here. Use the metabolite shadow price.
    """
    model = _build_toy_ec_model(enzyme_ub=ub, enzyme_coeff_in_R=coeff_in_R)
    _, coeffs = _fd_helper(model)  # finite-difference ground truth
    model.optimize()  # single solve
    shadow = abs(model.enzymes.get_by_id("E").shadow_price)
    assert coeffs[0] == pytest.approx(expected)
    assert shadow == pytest.approx(coeffs[0], rel=1e-6)


# --------------------------------------------------------------------------- #
# Thorough equivalence battery: shadow-price reproduces the finite-difference
# contract of get_conc_control_coeffs across realistic scenarios. (They are
# expected to disagree only when the 2x finite-difference step crosses an LP
# breakpoint -- documented at the bottom.)
# --------------------------------------------------------------------------- #

# Both implementations live in the production module after the swap; tests
# compare them as ground-truth helpers (the dispatcher uses shadow-price by
# default and falls back to finite-difference on scipy).
from geckopy.limit_proteins.get_conc_control_coeffs import (  # noqa: E402
    _finite_difference_coeffs as _fd_coeffs,
    _shadow_price_coeffs as _sp_coeffs,
)


def _shadow_helper(model, proteins=None, limit: float = 0.0):
    if proteins is None:
        proteins = list(model.ec.enzymes)
    return _sp_coeffs(model, proteins, limit)


def _fd_helper(
    model, proteins=None, limit: float = 0.0, fold_change: float = 2.0,
):
    if proteins is None:
        proteins = list(model.ec.enzymes)
    return _fd_coeffs(model, proteins, fold_change, limit)


def _build_two_enzyme_model(
    *, ub_E1: float = 5.0, ub_E2: float = 5.0,
    ex_lb: float = -1000.0,
) -> EcModel:
    """Two enzymes in series: A -> B (E1) -> C (E2), objective = SK_C.
    Either or both enzymes can be limiting depending on ``ub_E1``/``ub_E2``."""
    model = EcModel("toy2")
    A_e = cobra.Metabolite("A_e", compartment="e")
    A_c = cobra.Metabolite("A_c", compartment="c")
    B_c = cobra.Metabolite("B_c", compartment="c")
    C_c = cobra.Metabolite("C_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E1 = cobra.Metabolite("prot_E1", compartment="c")
    prot_E2 = cobra.Metabolite("prot_E2", compartment="c")
    model.add_metabolites([A_e, A_c, B_c, C_c, pool, prot_E1, prot_E2])

    EX_A = cobra.Reaction("EX_A"); EX_A.add_metabolites({A_e: -1.0})
    EX_A.lower_bound = ex_lb; EX_A.upper_bound = 0.0
    TR_A = cobra.Reaction("TR_A")
    TR_A.add_metabolites({A_e: -1.0, A_c: 1.0})
    TR_A.lower_bound = 0.0; TR_A.upper_bound = 1000.0
    R_AB = cobra.Reaction("R_AB")
    R_AB.add_metabolites({A_c: -1.0, B_c: 1.0, prot_E1: -1.0})
    R_AB.lower_bound = 0.0; R_AB.upper_bound = 1000.0
    R_BC = cobra.Reaction("R_BC")
    R_BC.add_metabolites({B_c: -1.0, C_c: 1.0, prot_E2: -1.0})
    R_BC.lower_bound = 0.0; R_BC.upper_bound = 1000.0
    SK_C = cobra.Reaction("SK_C")
    SK_C.add_metabolites({C_c: -1.0})
    SK_C.lower_bound = 0.0; SK_C.upper_bound = 1000.0
    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.lower_bound = 0.0; pool_ex.upper_bound = 1000.0
    u1 = cobra.Reaction("usage_prot_E1")
    u1.add_metabolites({pool: -1.0, prot_E1: 1.0})
    u1.lower_bound = 0.0; u1.upper_bound = ub_E1
    u2 = cobra.Reaction("usage_prot_E2")
    u2.add_metabolites({pool: -1.0, prot_E2: 1.0})
    u2.lower_bound = 0.0; u2.upper_bound = ub_E2

    model.add_reactions([EX_A, TR_A, R_AB, R_BC, SK_C, pool_ex, u1, u2])
    model.objective = "SK_C"
    model.ec = EcData(
        rxns=["R_AB", "R_BC"],
        kcat=np.array([1.0, 1.0]),
        source=["", ""], notes=["", ""], eccodes=["", ""],
        genes=["g_E1", "g_E2"], enzymes=["E1", "E2"],
        mw=np.array([100.0, 100.0]),
        sequence=["", ""], concs=np.array([ub_E1, ub_E2]),
        rxn_enz_mat=sparse.csr_matrix(np.eye(2)),
    )
    return model


@pytest.mark.parametrize(
    "ub, coeff_in_R", [(5.0, -1.0), (3.0, -1.0), (6.0, -2.0), (2.0, -3.0)],
)
def test_equiv_single_enzyme_default(ub, coeff_in_R):
    model = _build_toy_ec_model(enzyme_ub=ub, enzyme_coeff_in_R=coeff_in_R)
    fm, fc = _fd_helper(model)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)


def test_equiv_limit_param_skips_both():
    """A non-zero `limit` must filter the same proteins in both methods."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    # Cap upstream so usage/ub = 1/5 = 0.2.
    model.reactions.get_by_id("EX_A").lower_bound = -1.0
    fm, fc = _fd_helper(model, limit=0.5)
    sm, sc = _shadow_helper(model, limit=0.5)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)
    assert not fm.any()  # both correctly skip the under-used enzyme


def test_equiv_non_binding_enzyme():
    """Enzyme ub far above the substrate-limited optimum: usage is at the
    upstream limit (binding the substrate, not the enzyme). Both methods
    handle the relationship consistently."""
    model = _build_toy_ec_model(enzyme_ub=1000.0)
    model.reactions.get_by_id("EX_A").lower_bound = -10.0
    fm, fc = _fd_helper(model)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)


def test_equiv_two_enzymes_only_one_binding():
    """E1 is slack, E2 is the bottleneck; both methods agree on which is
    flagged and with what coefficient."""
    model = _build_two_enzyme_model(ub_E1=100.0, ub_E2=3.0)
    fm, fc = _fd_helper(model)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)


def test_equiv_subset_proteins():
    model = _build_two_enzyme_model(ub_E1=5.0, ub_E2=5.0)
    proteins = ["E2"]  # only ask about one
    fm, fc = _fd_helper(model, proteins=proteins)
    sm, sc = _shadow_helper(model, proteins=proteins)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)


def test_equiv_infeasible_model_both_zero():
    """An infeasible LP: both methods return all-False mask and zero coeffs."""
    model = _build_toy_ec_model(enzyme_ub=5.0)
    # Force infeasibility: require growth > what the model can deliver.
    sk = model.reactions.get_by_id("SK_B")
    sk.lower_bound = 1000.0  # impossible
    fm, fc = _fd_helper(model)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)
    assert not fm.any() and not sm.any()


@pytest.mark.parametrize(
    "solver",
    [
        "glpk",
        "glpk_exact",
        # optlang's scipy backend does not implement duals; the reduced-cost
        # function then falls back to the finite difference itself.
        "scipy",
        "gurobi",
    ],
)
def test_equiv_across_solvers(solver):
    """The reduced-cost/finite-difference equivalence holds for every LP
    solver, through the fallback where the solver exposes no duals. Uses the single-enzyme linear-regime model
    so the LP is non-degenerate / non-breakpoint (where duals would be
    non-unique)."""
    from cobra.util.solver import solvers
    if solver not in solvers:
        pytest.skip(f"{solver} not installed")
    model = _build_toy_ec_model(enzyme_ub=5.0)
    model.solver = solver
    fm, fc = _fd_helper(model)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(fm, sm)
    np.testing.assert_allclose(fc, sc, rtol=1e-6, atol=1e-12)


def test_finite_diff_underestimates_when_step_crosses_breakpoint():
    """Documents the one regime where the two methods DIVERGE: when the
    finite-difference 2x step pushes the LP through an active-set change
    (here, the substrate cap takes over from the enzyme cap), the
    finite-difference is the *averaged* slope over the step while the
    shadow price is the *local* marginal. The shadow price is therefore
    >= the finite-difference and is the analytically-correct local value."""
    # enzyme_ub=8, substrate cap=10. Doubling ub to 16 saturates at 10.
    model = _build_toy_ec_model(enzyme_ub=8.0)
    model.reactions.get_by_id("EX_A").lower_bound = -10.0
    _, fc = _fd_helper(model, fold_change=2.0)
    sm, sc = _shadow_helper(model)
    # finite-diff averages: (10 - 8) / (16 - 8) = 0.25.
    assert fc[0] == pytest.approx(0.25, rel=1e-6)
    # shadow price is the local marginal at the binding ub: 1.0.
    assert sc[0] == pytest.approx(1.0, rel=1e-6)
    assert sm[0] and sc[0] > fc[0]


# --------------------------------------------------------------------------- #
# Dispatcher: the public function defaults to shadow-price and falls back to
# finite-difference on solvers (scipy) that do not expose duals.
# --------------------------------------------------------------------------- #

def test_dispatcher_uses_shadow_price_on_default_solver():
    """On glpk (cobra default), the public function takes the shadow-price
    path. Verified by matching the shadow-price helper exactly (the
    finite-difference would diverge on a breakpoint scenario)."""
    model = _build_toy_ec_model(enzyme_ub=8.0)
    model.reactions.get_by_id("EX_A").lower_bound = -10.0  # breakpoint setup
    pm, pc = get_conc_control_coeffs(model, fold_change=2.0)
    sm, sc = _shadow_helper(model)
    np.testing.assert_array_equal(pm, sm)
    np.testing.assert_allclose(pc, sc, rtol=1e-6, atol=1e-12)
    # And it disagrees with the finite-difference at the breakpoint, proving
    # the shadow-price path was actually taken.
    _, fc = _fd_helper(model, fold_change=2.0)
    assert pc[0] != pytest.approx(fc[0])


def test_dispatcher_falls_back_to_finite_diff_on_scipy(caplog):
    """With the scipy solver (no LP duals), the dispatcher uses the
    finite-difference path and reports the fallback once."""
    from cobra.util.solver import solvers
    if "scipy" not in solvers:
        pytest.skip("scipy not installed")
    import importlib
    import logging
    # Use importlib because the submodule name shadows the same-named
    # function re-exported on `geckopy.limit_proteins`.
    mod_ccc = importlib.import_module(
        "geckopy.limit_proteins.get_conc_control_coeffs"
    )
    # Reset the one-shot guard so the info log fires for this test.
    mod_ccc._FALLBACK_REPORTED.clear()

    model = _build_toy_ec_model(enzyme_ub=8.0)
    model.reactions.get_by_id("EX_A").lower_bound = -10.0
    model.solver = "scipy"

    with caplog.at_level(logging.INFO, logger=mod_ccc.__name__):
        pm, pc = get_conc_control_coeffs(model, fold_change=2.0)
    fm, fc = _fd_helper(model, fold_change=2.0)
    np.testing.assert_array_equal(pm, fm)
    np.testing.assert_allclose(pc, fc, rtol=1e-6, atol=1e-12)
    assert "does not expose LP duals" in caplog.text
