"""Smoke tests that the deprecated function aliases still work.

After the rename round, each old function name lives as a thin
``DeprecationWarning``-emitting wrapper around the new canonical
name. These tests just confirm the alias resolves, forwards
arguments, and emits the expected warning class. Behavioural
correctness is covered by the renamed canonical-name tests.
"""
import warnings


def _assert_deprecated(callable_, *args, expected_substring: str, **kwargs):
    """Call ``callable_`` and assert a DeprecationWarning containing
    ``expected_substring`` was raised."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        callable_(*args, **kwargs)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and expected_substring in str(w.message)
        for w in recorded
    ), (
        f"Expected DeprecationWarning mentioning {expected_substring!r}; "
        f"got: {[str(w.message) for w in recorded]}"
    )


def test_get_ec_from_gem_alias_emits_warning():
    from geckopy.get_enzyme_data import get_ec_from_gem
    from geckopy.ec_model.ec_data import EcData
    from geckopy import EcModel

    # An EcModel with no ec.rxns is a valid no-op input; the function
    # short-circuits before touching anything, so we can use it to
    # exercise the alias-warning path without setting up a full model.
    model = EcModel("empty")
    model.ec = EcData()
    _assert_deprecated(
        get_ec_from_gem, model,
        expected_substring="fill_eccodes_from_gem",
    )


def test_select_kcat_value_alias_emits_warning():
    import pandas as pd

    from geckopy.gather_kcats import select_kcat_value
    from geckopy.ec_model.ec_data import EcData
    from geckopy import EcModel

    model = EcModel("empty")
    model.ec = EcData()
    empty_kcat_list = pd.DataFrame(
        columns=["rxn_id", "kcat", "wildcard_level", "origin"],
    )
    # The underlying call may fail on a malformed kcat_list (missing
    # columns, etc.); we only care that the alias emits the warning
    # before delegating, not that the delegated call succeeds.
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            select_kcat_value(model, empty_kcat_list)
        except Exception:
            pass
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "apply_kcat_list" in str(w.message)
        for w in recorded
    )


def test_constrain_flux_data_alias_emits_warning():
    from pathlib import Path

    from geckopy.databases import load_flux_data
    from geckopy.limit_proteins import constrain_flux_data

    # We need a model + flux_data pair. Use a stub model and a flux-data
    # file from the test fixtures; the call may raise an inner error
    # (the alias only checks for the deprecation warning, not for the
    # success of the underlying call), so wrap accordingly.
    flux_data_path = (
        Path(__file__).parents[1]
        / "tutorials" / "full_ecModel" / "data" / "fluxData.tsv"
    )
    if not flux_data_path.is_file():
        import pytest
        pytest.skip("flux_data fixture not present")
    flux_data = load_flux_data(flux_data_path)

    class _MockModel:
        adapter = None

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            constrain_flux_data(_MockModel(), flux_data)
        except (AttributeError, ValueError):
            pass  # underlying call may fail on the mock; we only test the warning
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "apply_flux_data_constraints" in str(w.message)
        for w in recorded
    )


def test_sigma_fitter_alias_emits_warning():
    from geckopy.kcat_tuning import sigma_fitter

    class _MockModel:
        pass

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            sigma_fitter(_MockModel())
        except Exception:
            pass
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "fit_sigma" in str(w.message)
        for w in recorded
    )


def test_get_standard_kcat_alias_emits_warning():
    """``get_standard_kcat`` (deprecated) -> ``assign_standard_kcat``."""
    from geckopy.gather_kcats import get_standard_kcat

    class _MockModel:
        adapter = None

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            get_standard_kcat(_MockModel(), None)
        except Exception:
            pass
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "assign_standard_kcat" in str(w.message)
        for w in recorded
    )


def test_get_ec_from_database_alias_emits_warning():
    from geckopy.get_enzyme_data import get_ec_from_database

    class _MockModel:
        adapter = None
        ec = None

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            get_ec_from_database(_MockModel(), None)
        except Exception:
            pass
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "fill_eccodes_from_database" in str(w.message)
        for w in recorded
    )


def test_merge_dlkcat_and_fuzzy_kcats_alias_emits_warning():
    """``merge_dlkcat_and_fuzzy_kcats`` (deprecated) -> ``merge_kcats``."""
    from geckopy.gather_kcats import merge_dlkcat_and_fuzzy_kcats

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            merge_dlkcat_and_fuzzy_kcats(None, None)
        except Exception:
            pass
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "merge_kcats" in str(w.message)
        for w in recorded
    )


def test_sigma_fitter_alias_passes_method(monkeypatch):
    import importlib

    module = importlib.import_module("geckopy.kcat_tuning.sigma_fitter")
    seen = {}
    monkeypatch.setattr(module, "fit_sigma", lambda model, **kwargs: seen.update(kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module.sigma_fitter(object(), method="bisect")
    assert seen["method"] == "bisect"
