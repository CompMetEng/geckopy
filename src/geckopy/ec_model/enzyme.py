"""A friendly accessor for individual enzymes in an EcModel.

Direct access to enzyme data through ``model.ec`` works but is
awkward — you have to look up the right index, edit the right
array, and remember to re-run ``apply_kcat_constraints`` to push
your change into the LP.

This module provides a nicer surface. ``model.enzymes`` is an
``EnzymeView`` over all enzymes in the model. Each enzyme is
reachable as an ``Enzyme`` proxy:

.. code-block:: python

    enz = model.enzymes.get_by_id("P00350")
    enz.mw                      # read MW (Da)
    enz.concentration = 1e-3    # write proteomics conc (mg/gDCW)
    enz.kcats["R_FOO"] = 30.0   # write a new kcat for one reaction
    enz.shadow_price            # dual after a solve

The proxy is stateless: it holds no cached data of its own. Every
read and write goes through ``model.ec`` and the underlying cobra
model. Don't cache an ``Enzyme`` instance long-term; rebuild it
via ``model.enzymes.get_by_id(...)`` when you need it.

Ported from the legacy geckopy package described in Carrasco et al.
(2023, https://doi.org/10.1128/spectrum.01705-23), file
geckopy/protein.py:43-454 (Protein class). Re-implemented as a
proxy rather than a ``cobra.Object`` subclass because the new
package represents enzymes as ``prot_<id>`` metabolites plus
``usage_prot_<id>`` reactions, so the proxy just forwards to
those.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import cobra

from ..solver_support import constraint_shadow_prices
from .constants import PROT_PREFIX, USAGE_PREFIX
from .pipeline.populate_ec import split_light_rxn_id

if TYPE_CHECKING:
    from .ec_model import EcModel


class Enzyme:
    """Stateless proxy for one enzyme in an EcModel.

    All reads and writes go through ``model.ec`` and the underlying
    cobra model. Do not cache instances; rebuild via
    ``model.enzymes.get_by_id(uniprot)``.
    """

    __slots__ = ("_model", "_uniprot")

    def __init__(self, model: "EcModel", uniprot_id: str) -> None:
        if uniprot_id not in model.ec.enzymes:
            raise KeyError(f"Enzyme {uniprot_id!r} not in model.ec.enzymes")
        self._model = model
        self._uniprot = uniprot_id

    # ---- identity ----
    @property
    def id(self) -> str:
        return self._uniprot

    @property
    def index(self) -> int:
        return self._model.ec.enzymes.index(self._uniprot)

    @property
    def prot_metabolite_id(self) -> str:
        return f"{PROT_PREFIX}{self._uniprot}"

    @property
    def usage_reaction_id(self) -> str:
        return f"{USAGE_PREFIX}{self._uniprot}"

    @property
    def prot_metabolite(self) -> cobra.Metabolite:
        self._require_full_model("prot_metabolite")
        return self._model.metabolites.get_by_id(self.prot_metabolite_id)

    @property
    def usage_reaction(self) -> cobra.Reaction:
        self._require_full_model("usage_reaction")
        return self._model.reactions.get_by_id(self.usage_reaction_id)

    def _require_full_model(self, attr_name: str) -> None:
        """Raise NotImplementedError if the model is a light ecModel.

        Light models do not have per-enzyme ``prot_<id>`` metabolites
        or ``usage_prot_<id>`` reactions, so the proxy attributes
        that depend on those (``flux``, ``cap_usage``, ``upper_bound``,
        ``shadow_price``, ``prot_metabolite``, ``usage_reaction``)
        are unavailable on light models. Read-only metadata
        (``mw``, ``gene``, ``sequence``, ``concentration``, ``kcats``)
        works in both layouts because it reads from ``model.ec.*``
        arrays that exist in both.
        """
        if self._model.ec.gecko_light:
            raise NotImplementedError(
                f"Enzyme.{attr_name} is unavailable for gecko-light "
                f"models (light models have no per-enzyme prot/usage "
                f"machinery). Read-only metadata (mw, gene, sequence, "
                f"concentration, kcats) is available."
            )

    # ---- per-enzyme scalar data (live indexing into model.ec) ----
    @property
    def gene(self) -> str:
        return self._model.ec.genes[self.index]

    @property
    def sequence(self) -> str:
        return self._model.ec.sequence[self.index]

    @property
    def mw(self) -> float:
        """Molecular weight in Da."""
        return float(self._model.ec.mw[self.index])

    @mw.setter
    def mw(self, value: float) -> None:
        """Set MW (Da). Re-applies kcat constraints for every reaction
        that uses this enzyme, since the LP coefficient depends on MW.
        """
        from .pipeline.apply_kcat import apply_kcat_constraints
        self._model.ec.mw[self.index] = float(value)
        col = self._model.ec.rxn_enz_mat.tocsc().getcol(self.index)
        ec_rxn_ids = [self._model.ec.rxns[i] for i in col.nonzero()[0]]
        if ec_rxn_ids:
            apply_kcat_constraints(self._model, update_rxns=ec_rxn_ids)

    @property
    def concentration(self) -> float:
        """Proteomics-measured concentration in mg/gDCW. NaN if unmeasured."""
        return float(self._model.ec.concs[self.index])

    @concentration.setter
    def concentration(self, value: float) -> None:
        """Set the proteomics concentration and update the usage upper bound.

        Passing ``np.nan`` returns the enzyme to the shared pool
        (resets the upper-bound constraint set by
        ``constrain_enz_concs`` to the default 1000).
        """
        if self._model.ec.gecko_light:
            raise NotImplementedError(
                "concentration setter not supported in gecko-light"
            )
        from ..limit_proteins.constrain_enz_concs import constrain_enz_concs
        self._model.ec.concs[self.index] = float(value)
        constrain_enz_concs(self._model, restrict_to=[self._uniprot])

    # ---- solver-side reads ----
    @property
    def flux(self) -> float:
        """Primal flux of usage_prot_<id> from the last solve (mg/gDCW).

        Raises RuntimeError if no solution is cached.
        """
        rxn = self.usage_reaction
        try:
            return float(rxn.flux)
        except Exception as e:  # cobra raises OptimizationError, AttributeError, ...
            raise RuntimeError(
                f"No solution available for {self._uniprot}"
            ) from e

    @property
    def cap_usage(self) -> float:
        """flux / upper_bound; NaN if upper_bound == 0."""
        ub = self.upper_bound
        if ub == 0:
            return float("nan")
        return abs(self.flux) / ub

    @property
    def upper_bound(self) -> float:
        return float(self.usage_reaction.upper_bound)

    @upper_bound.setter
    def upper_bound(self, value: float) -> None:
        self.usage_reaction.upper_bound = float(value)

    @property
    def shadow_price(self) -> float:
        """Dual of the prot_<id> mass-balance constraint, on every solver (scipy included)."""
        self._require_full_model("shadow_price")
        return float(constraint_shadow_prices(self._model, [self.prot_metabolite_id])[0])

    @property
    def reactions(self) -> frozenset:
        """Metabolic reactions catalysed by this enzyme.

        Defined as the non-zero entries in
        ``rxn_enz_mat[:, self.index]``, intersected with reactions
        still present in the cobra model.

        For gecko-light models, each ec row carries a ``###_`` prefix
        identifying its isozyme; the prefix is stripped before the cobra
        lookup so two isozyme rows of the same cobra reaction return a
        single ``cobra.Reaction``.
        """
        col = self._model.ec.rxn_enz_mat.tocsc().getcol(self.index)
        rxn_indices = col.nonzero()[0]
        is_light = self._model.ec.gecko_light
        seen_cobra_ids: set[str] = set()
        rxns: list[cobra.Reaction] = []
        for i in rxn_indices:
            ec_rxn_id = self._model.ec.rxns[i]
            cobra_id = (
                split_light_rxn_id(ec_rxn_id)[1] if is_light else ec_rxn_id
            )
            if cobra_id in seen_cobra_ids:
                continue
            seen_cobra_ids.add(cobra_id)
            try:
                rxns.append(self._model.reactions.get_by_id(cobra_id))
            except KeyError:
                continue
        return frozenset(rxns)

    @property
    def kcats(self) -> "Kcats":
        return Kcats(self)

    # ---- repr ----
    def __repr__(self) -> str:
        return f"<Enzyme {self._uniprot} gene={self.gene} mw={self.mw:.1f}Da>"

    def _repr_html_(self) -> str:
        # NotImplementedError covers gecko-light (no usage reaction);
        # RuntimeError covers full models that haven't been solved yet.
        try:
            flux_str = f"{self.flux:.3g}"
            cap_str = f"{self.cap_usage:.3g}"
        except (RuntimeError, NotImplementedError):
            flux_str = "-"
            cap_str = "-"
        return (
            f"<table>"
            f"<tr><td><strong>Enzyme</strong></td><td>{self._uniprot}</td></tr>"
            f"<tr><td>Gene</td><td>{self.gene}</td></tr>"
            f"<tr><td>MW (Da)</td><td>{self.mw:.1f}</td></tr>"
            f"<tr><td>Concentration (mg/gDCW)</td>"
            f"<td>{self.concentration:.3g}</td></tr>"
            f"<tr><td>Flux</td><td>{flux_str}</td></tr>"
            f"<tr><td>Capacity usage</td><td>{cap_str}</td></tr>"
            f"<tr><td>Reactions</td><td>{len(self.reactions)}</td></tr>"
            f"</table>"
        )


class Kcats:
    """Dict-like view of ``{reaction_id: kcat_per_s}`` for one enzyme.

    Reads return ``model.ec.kcat[rxn_idx]`` for any reaction in which
    this enzyme participates (non-zero column in ``rxn_enz_mat``).

    Writes update ``model.ec.kcat`` for the cell, then call
    ``apply_kcat_constraints`` for that reaction. Raises
    ``ValueError`` if the target reaction has more than one enzyme:
    per-enzyme kcats are ambiguous in that case; the user must edit
    ``model.ec.kcat[idx]`` and call ``apply_kcat_constraints``
    themselves.
    """

    __slots__ = ("_enzyme",)

    def __init__(self, enzyme: "Enzyme") -> None:
        self._enzyme = enzyme

    def _rxn_indices(self) -> list[int]:
        col = (
            self._enzyme._model.ec.rxn_enz_mat
            .tocsc()
            .getcol(self._enzyme.index)
        )
        return list(col.nonzero()[0])

    def __iter__(self) -> Iterator[str]:
        rxns = self._enzyme._model.ec.rxns
        for i in self._rxn_indices():
            yield rxns[i]

    def __len__(self) -> int:
        return len(self._rxn_indices())

    def __contains__(self, rxn_id: str) -> bool:
        return rxn_id in set(self)

    def __getitem__(self, rxn_id: str) -> float:
        if rxn_id not in self:
            raise KeyError(f"{rxn_id} not catalysed by {self._enzyme.id}")
        idx = self._enzyme._model.ec.rxns.index(rxn_id)
        return float(self._enzyme._model.ec.kcat[idx])

    def __setitem__(self, rxn_id: str, kcat_per_s: float) -> None:
        from .pipeline.apply_kcat import apply_kcat_constraints
        model = self._enzyme._model
        if model.ec.gecko_light:
            raise NotImplementedError(
                "kcats setter not supported in gecko-light"
            )
        if rxn_id not in self:
            raise KeyError(
                f"{rxn_id} not catalysed by {self._enzyme.id}"
            )
        rxn_idx = model.ec.rxns.index(rxn_id)
        n_enzymes_on_rxn = model.ec.rxn_enz_mat.getrow(rxn_idx).nnz
        if n_enzymes_on_rxn > 1:
            raise ValueError(
                f"Reaction {rxn_id} is catalysed by {n_enzymes_on_rxn} "
                "enzymes; per-enzyme kcat is ambiguous. Set "
                "model.ec.kcat[idx] directly and call "
                "apply_kcat_constraints."
            )
        model.ec.kcat[rxn_idx] = float(kcat_per_s)
        apply_kcat_constraints(model, update_rxns=[rxn_id])

    def keys(self):
        return list(self)

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={self[k]:.3g}" for k in self)
        return f"Kcats({{{items}}})"


class EnzymeView:
    """Lazy DictList-like view over EcModel enzymes.

    Always reflects live state of ``model.ec.enzymes``.
    """

    __slots__ = ("_model",)

    def __init__(self, model: "EcModel") -> None:
        self._model = model

    def __iter__(self) -> Iterator[Enzyme]:
        for u in self._model.ec.enzymes:
            yield Enzyme(self._model, u)

    def __len__(self) -> int:
        return len(self._model.ec.enzymes)

    def __contains__(self, key) -> bool:
        if isinstance(key, Enzyme):
            return key.id in self._model.ec.enzymes
        return key in self._model.ec.enzymes

    def __getitem__(self, key) -> Enzyme:
        if isinstance(key, int):
            return Enzyme(self._model, self._model.ec.enzymes[key])
        return self.get_by_id(key)

    def get_by_id(self, uniprot_id: str) -> Enzyme:
        return Enzyme(self._model, uniprot_id)

    def query(self, predicate, attribute: str = "id") -> list[Enzyme]:
        """Filter enzymes by a predicate on the named attribute.

        ``attribute`` selects which per-enzyme value (default ``"id"``)
        is passed to ``predicate``. Returns the list of ``Enzyme``
        objects for which the predicate returns true.

        Examples
        --------
        >>> model.enzymes.query(lambda gid: gid.startswith("b0"), "gene")
        >>> model.enzymes.query(lambda mw: mw > 50000, "mw")
        """
        result = []
        for enz in self:
            value = getattr(enz, attribute)
            if predicate(value):
                result.append(enz)
        return result
