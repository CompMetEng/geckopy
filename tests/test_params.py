"""Tests for adapter parameter schemas (validation)."""
import pytest
from pydantic import ValidationError

from geckopy.adapter.params import EvotuneParams, SourceGroupRule


def test_default_groups_are_custom_brenda_ec_tree_combined():
    groups = EvotuneParams().source_groups
    assert {k: (v.sources, v.sigma0_log) for k, v in groups.items()} == {
        "custom": (["custom"], 0.1),
        "brenda": (["brenda"], 0.2),
        "ec_tree": (["ec_tree"], 0.3),
        "combined": (["combined"], 0.4),
    }
    assert EvotuneParams().sigma0_log_default == 0.5


def test_groups_given_replace_the_defaults():
    bp = EvotuneParams(source_groups={"a": SourceGroupRule(sources=["src_a"], sigma0_log=0.1)})
    assert set(bp.source_groups) == {"a"}


@pytest.mark.parametrize("kwargs", [{"sources": ["a"]}, {"sigma0_log": 0.1}])
def test_a_group_needs_sources_and_sigma0_log(kwargs):
    with pytest.raises(ValidationError):
        SourceGroupRule(**kwargs)


def test_dropped_fields_are_rejected():
    """Removed settings are unknown keys, not silently ignored."""
    for kwargs in ({"target_accept": 10.0}, {"variance_cap_default": 10.0}, {"sigma0_log_source": {}}):
        with pytest.raises(ValidationError):
            EvotuneParams(**kwargs)
    with pytest.raises(ValidationError):
        SourceGroupRule(sources=["a"], sigma0_log=0.1, match_okp=True)
