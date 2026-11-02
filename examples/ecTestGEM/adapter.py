"""Custom adapter for the ecTestGEM test model.

Port of test/unit_tests/ecTestGEM/TestGEMAdapter.m from GECKO MATLAB.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from geckopy import ModelAdapter

if TYPE_CHECKING:
    import cobra


class TestGEMAdapter(ModelAdapter):
    """Test GEM adapter. Marks R4 as the sole spontaneous reaction."""

    def get_spontaneous_reactions(self, model: "cobra.Model") -> list[str]:
        # In MATLAB TestGEMAdapter, spont(5) = true marks R4 as spontaneous.
        # Using the reaction ID rather than the index is more robust to
        # reordering by cobrapy's SBML parser.
        if "R4" in {r.id for r in model.reactions}:
            return ["R4"]
        return []
