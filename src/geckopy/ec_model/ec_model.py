"""``EcModel``: a regular ``cobra.Model`` extended with enzyme data.

An ``EcModel`` *is* a cobra model (it subclasses ``cobra.Model``),
so everything you know how to do with cobra still works: solve
FBAs, edit reactions, inspect metabolites, etc.

It carries two extra attributes on top of cobra:

- ``ec``: an ``EcData`` holding the enzyme-related arrays (kcats,
  enzyme MW, the reaction-enzyme coupling matrix, ...).
- ``adapter``: the ``ModelAdapter`` used to build it, so downstream
  helper functions can reach organism-specific parameters
  (taxonomy id, biomass reaction, ...) without being passed
  the adapter explicitly every time.

There is also an ``enzymes`` view on every EcModel that lets you
write ``model.enzymes.get_by_id("P00350").kcats["R_FOO"] = 30.0``
to update an enzyme-level constraint through a friendly proxy.
See ``geckopy.ec_model.enzyme.Enzyme`` for the full surface.
"""
from __future__ import annotations

import copy as _copy
from typing import TYPE_CHECKING, Optional, Union

import cobra

from ..solver_support import is_scipy_interface, repair_scipy_problem, switch_to_scipy
from .ec_data import EcData

if TYPE_CHECKING:
    from ..adapter import ModelAdapter


class EcModel(cobra.Model):
    """A cobra model with enzyme-constraint data attached.

    Subclasses ``cobra.Model`` so anything you do with a regular
    cobra model still works. Adds two attributes:

    Attributes
    ----------
    ec : EcData
        The enzyme-constraint data: kcats, enzyme MW, sequence,
        proteomics concentrations, and the sparse matrix saying
        which enzymes catalyse which reactions. See
        ``geckopy.ec_model.ec_data.EcData``.
    adapter : ModelAdapter or None
        The adapter the ecModel was built with (organism
        parameters, paths, etc.). ``None`` if the model was loaded
        from disk and no adapter was attached at load time.
    enzymes : EnzymeView
        Lazy view that lets you reach individual enzymes via
        ``model.enzymes.get_by_id("P00350")``. Each call returns a
        live ``Enzyme`` proxy; reads and writes always go through
        the underlying ``ec`` data.
    """

    def __init__(
        self,
        id_or_model: Optional[Union[str, cobra.Model]] = None,
        name: Optional[str] = None,
        *,
        adapter: Optional["ModelAdapter"] = None,
        gecko_light: bool = False,
    ):
        super().__init__(id_or_model, name)
        self.ec: EcData = EcData(gecko_light=gecko_light)
        self.adapter: Optional["ModelAdapter"] = adapter
        # Deferred import: enzyme.py uses EcModel only under TYPE_CHECKING,
        # but EnzymeView(self) needs the class at runtime.
        from .enzyme import EnzymeView
        self.enzymes = EnzymeView(self)

    @property
    def solver(self):
        """The optlang model. With the scipy interface, reading it also repairs the
        problem's variable index map (see :mod:`geckopy.solver_support`), so values read
        after any removal of variables come from the right columns."""
        repair_scipy_problem(self)
        return cobra.Model.solver.fget(self)

    @solver.setter
    def solver(self, value) -> None:
        """Set the solver. Moving a model to the scipy interface goes through
        :func:`geckopy.solver_support.switch_to_scipy`: optlang's clone grows the dense
        coefficient matrix row by row, which takes hours for a genome-scale model."""
        if not self._contexts and is_scipy_interface(value) and not is_scipy_interface(self.problem):
            switch_to_scipy(self)
        else:
            cobra.Model.solver.fset(self, value)

    @classmethod
    def from_cobra(
        cls,
        model: cobra.Model,
        adapter: Optional["ModelAdapter"] = None,
        *,
        gecko_light: bool = False,
    ) -> "EcModel":
        """Wrap an existing cobra.Model as an EcModel.

        ``adapter`` is optional: pass ``None`` to wrap a model for
        inspection without a project (downstream functions that need
        organism parameters will then require an explicit ``adapter``).

        Does not yet run makeEcModel; that is a separate pipeline step
        that mutates the wrapped model and populates `self.ec`.
        """
        ec_model = cls(model, adapter=adapter, gecko_light=gecko_light)
        return ec_model

    def copy(self) -> "EcModel":
        """Copy the model, including an independent ``ec`` substructure.

        ``cobra.Model.copy`` rebuilds the reactions, metabolites and genes
        but carries every other attribute over **by reference**, so without
        this override ``model.copy().ec`` would be the same ``EcData``
        object as ``model.ec``: writing a kcat on the copy would silently
        change the original. For the same reason the copy's ``enzymes``
        view would still be bound to the model it was copied from.

        This override clones ``ec`` and rebinds ``enzymes`` to the new
        model, so the copy is fully independent.

        ``adapter`` stays a shared reference: it is immutable project
        configuration, not model state, and copying it would detach the
        copy from the project it belongs to.
        """
        new = super().copy()
        # deepcopy rather than a field-by-field clone: EcData holds only
        # lists, numpy arrays, a scipy sparse matrix and a bool, and this
        # keeps working if raven-toolbox adds a field.
        new.ec = _copy.deepcopy(self.ec)
        new.adapter = self.adapter
        # Deferred import, as in __init__: enzyme.py imports EcModel only
        # under TYPE_CHECKING.
        from .enzyme import EnzymeView
        new.enzymes = EnzymeView(new)
        return new

    def validate_ec(self) -> None:
        """Validate internal consistency of the ec substructure.

        Runs ``EcData.validate`` for the array-shape checks, then
        confirms every reaction id in ``ec.rxns`` exists in the model;
        raises ``ValueError`` (listing a few example ids) if any are
        missing.
        """
        self.ec.validate()
        unknown = set(self.ec.rxns) - {r.id for r in self.reactions}
        if unknown:
            preview = sorted(unknown)[:5]
            raise ValueError(
                f"{len(unknown)} reaction IDs in ec.rxns are not present "
                f"in the model (examples: {preview})"
            )
