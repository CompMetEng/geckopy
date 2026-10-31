"""Flux-Scanning with Enforced Objective Function for an ecModel.

Thin wrapper around :func:`raven_toolbox.analysis.fseof.fseof` that adds
ec-specific concerns on top of raven's general-purpose FSEOF:

- resolves the biomass reaction from ``model.adapter`` when not passed
  explicitly;
- optionally runs a carbon-source consistency check (warns when the
  carbon-source lower bound is tighter than its uptake at biomass-max);
- anchors the scan's floor to the production-target reaction's flux at
  biomass-optimum (matching ``ecFSEOF.m``'s ``iniTarget``), rather than
  raven-toolbox's default fixed-fraction-of-maximum floor;
- restricts targets (and the reported scan) to gene-associated
  reactions whose GPR doesn't reference the "standard" placeholder
  pseudogene, matching ``ecFSEOF.m``'s candidate-set restriction;
- fixes the target flux at each level (lower and upper bound), as
  ``ecFSEOF.m`` does;
- orders ``targets`` as ``ecFSEOF.m`` returns them: reaction targets first,
  then transport targets, each group sorted by score, with a ``transport``
  boolean column that marks the group;
- filters ``usage_prot_*`` rows out of the returned ``scan`` /
  ``targets`` DataFrames — those reactions are protein-pool plumbing,
  never engineering targets.

The selection method and output shape come from raven-toolbox: each
reaction's flux is regressed against the enforced product flux across
the scan, and a reaction is a target if ``|correlation| >=
correlation_threshold`` and ``|slope| >= flux_eps``. Targets are
classified as ``amplify`` / ``knockdown`` / ``knockout``. See
:class:`raven_toolbox.analysis.fseof.FSEOFResult` for the full output
schema.

Ported from GECKO MATLAB: ``src/geckomat/utilities/ecFSEOF.m``,
re-designed around :func:`raven_toolbox.analysis.fseof.fseof`.

Callers who want the result tables on disk can call
``result.targets.to_csv(...)`` themselves. Per-gene essentiality is
not computed here; use :func:`cobra.flux_analysis.single_gene_deletion`,
which works directly on ecModels since GPRs already gate the catalysed
reactions. Targets come back as one table, reaction targets before transport
targets (``ecFSEOF.m``'s ``rxnTargets`` and ``transportTargets``); the
``transport`` column tells them apart.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from raven_toolbox.analysis.fseof import FSEOFResult, fseof
from ..solver_support import optimize_model

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)

_USAGE_PROT_PREFIX = "usage_prot_"
_STANDARD_GENE = "standard"


def ec_fseof(
    model: "EcModel",
    prod_target_rxn: str,
    cs_rxn: Optional[str] = None,
    *,
    n_steps: int = 10,
    bio_rxn: Optional[str] = None,
    max_fraction: float = 0.9,
    correlation_threshold: float = 0.9,
    flux_eps: float = 1e-6,
) -> FSEOFResult:
    """Run Flux-Scanning with Enforced Objective Function on an ecModel.

    Enforces an increasing flux toward ``prod_target_rxn`` while
    optimising biomass, runs pFBA at each step, regresses every
    reaction's flux against the enforced product flux, and returns
    reactions with strong slope-vs-correlation signal as engineering
    targets.

    Parameters
    ----------
    model
        EcModel with the protein-pool / usage-rxn machinery.
    prod_target_rxn
        Reaction id for the production target (typically a
        product-exchange).
    cs_rxn
        Reaction id for the main carbon source. When passed, a warning
        is emitted if its lower bound is tighter than its uptake at
        biomass-max (a common mis-configuration). ``None`` skips the
        check.
    n_steps
        Number of enforced-flux levels in the scan. Default 10.
    bio_rxn
        Biomass reaction id. Defaults to ``adapter.params.bio_rxn`` when
        ``model.adapter`` is set.
    max_fraction
        Top of the scan range as a fraction of the theoretical maximum
        product flux. Default 0.9. The floor of the scan range is the
        production target's flux at biomass-optimum, not a fraction of
        this ceiling.
    correlation_threshold
        A reaction is a target when ``|corr(flux, enforced)| >=`` this
        value. Default 0.9.
    flux_eps
        Minimum ``|slope|`` (flux change per enforced-flux step) for
        a reaction to count as responsive; below this it is treated
        as flat regardless of correlation, so it cannot become a
        target. Default 1e-6.

    Returns
    -------
    FSEOFResult
        raven-toolbox's FSEOF output dataclass, with ``usage_prot_*``
        rows filtered out of both ``scan`` and ``targets``. ``targets``
        has an added boolean column ``transport`` and lists the reaction
        targets first, then the transport targets, each group in the
        order of the score ``|slope| * |correlation|`` (equal scores in
        model order); a reaction is a transport target when a metabolite
        name occurs more than once among its metabolites, not counting
        the ``prot_*`` ones. A caller that takes the top N rows gets the
        top reaction targets, then transport targets. See
        :class:`raven_toolbox.analysis.fseof.FSEOFResult` for the full
        field list (``scan``, ``enforced``, ``targets``, plus the
        ``amplification`` / ``knockout`` / ``gene_targets`` properties).
    """
    from ..adapter import resolve_param
    bio_rxn_id = resolve_param(
        model, bio_rxn, "bio_rxn",
        purpose="ec_fseof needs the biomass reaction id",
    )

    if cs_rxn is not None:
        _check_cs_consistency(model, bio_rxn_id, cs_rxn)

    min_target = _biomass_optimal_target_flux(model, bio_rxn_id, prod_target_rxn)

    result = fseof(
        model, prod_target_rxn,
        biomass_rxn=bio_rxn_id,
        n_steps=n_steps,
        max_fraction=max_fraction,
        correlation_threshold=correlation_threshold,
        flux_eps=flux_eps,
        min_target=min_target,
        equality=True,
    )
    result = _drop_usage_prot(result)
    result = _drop_ungated_reactions(result, model)
    return _group_transport_targets(result, model)


def _check_cs_consistency(
    model: "EcModel", bio_rxn_id: str, cs_rxn_id: str,
) -> None:
    """Warn when the carbon source can't deliver its biomass-max uptake.

    If the carbon-source reaction's lower bound is less negative than
    its biomass-max flux, the scan will silently flatten at the bound
    and the slopes mean less than they look.
    """
    cs_rxn_obj = model.reactions.get_by_id(cs_rxn_id)
    with model:
        model.objective = bio_rxn_id
        sol = optimize_model(model)
        if sol.status != "optimal":
            return
        cs_flux = float(sol.fluxes.get(cs_rxn_id, 0.0))
    if cs_rxn_obj.lower_bound < cs_flux:
        logger.warning(
            "ec_fseof: carbon source %r lower bound is %g but uptake "
            "at biomass-max is %g; consider tightening so the scan "
            "doesn't flatten at the bound.",
            cs_rxn_id, cs_rxn_obj.lower_bound, cs_flux,
        )


def _biomass_optimal_target_flux(
    model: "EcModel", bio_rxn_id: str, prod_target_rxn_id: str,
) -> float:
    """The production-target reaction's flux at biomass-optimum.

    Matches ``ecFSEOF.m``'s ``iniTarget``: the scan's floor is where the
    model actually sits before any production is enforced, which can be
    near zero for a target with no baseline flux -- not a fixed fraction
    of the theoretical maximum.
    """
    with model:
        model.objective = bio_rxn_id
        sol = optimize_model(model)
        if sol.status != "optimal":
            return 0.0
        return float(sol.fluxes.get(prod_target_rxn_id, 0.0))


def _gated_reaction(model: "EcModel", rxn_id: str) -> bool:
    """Whether a reaction can be an engineering target: it must have a
    GPR, and that GPR must not reference the "standard" placeholder
    pseudogene added by :func:`assign_standard_kcat` for reactions with
    no real enzyme assignment."""
    gpr = model.reactions.get_by_id(rxn_id).gene_reaction_rule
    return bool(gpr) and _STANDARD_GENE not in gpr


def _drop_ungated_reactions(result: FSEOFResult, model: "EcModel") -> FSEOFResult:
    """Restrict ``scan`` and ``targets`` to gene-associated, non-standard
    reactions, matching ``ecFSEOF.m``'s candidate-set restriction (a
    reaction with no GPR, or only the ec.kcat-standard-value pseudogene,
    can't be engineered)."""
    scan = result.scan[[_gated_reaction(model, rid) for rid in result.scan.index]]
    targets = result.targets[
        result.targets["reaction"].map(lambda rid: _gated_reaction(model, rid))
    ].reset_index(drop=True)
    return FSEOFResult(scan=scan, enforced=result.enforced, targets=targets)


def _group_transport_targets(result: FSEOFResult, model: "EcModel") -> FSEOFResult:
    """Add the ``transport`` column and list reaction targets before transport targets.

    As in ``ecFSEOF.m``, a reaction is a transport reaction when a metabolite name repeats
    among its metabolites after the ``prot_*`` metabolites are dropped. The order within a
    group is the order of ``result.targets``.
    """
    targets = result.targets.copy()
    targets["transport"] = [
        _is_transport(model.reactions.get_by_id(rid)) for rid in targets["reaction"]
    ]
    targets = targets.sort_values("transport", kind="stable", ignore_index=True)
    return FSEOFResult(scan=result.scan, enforced=result.enforced, targets=targets)


def _is_transport(reaction) -> bool:
    names = [m.name for m in reaction.metabolites if not (m.name or "").startswith("prot_")]
    return len(names) != len(set(names))


def _drop_usage_prot(result: FSEOFResult) -> FSEOFResult:
    """Strip ``usage_prot_*`` rows from raven's ``scan`` and ``targets``.

    These reactions exist solely to draw the per-enzyme prot mets out of
    the shared pool; they have no metabolic meaning of their own and
    can't be engineered. Filtering at the boundary keeps the rest of
    raven's output structure intact (the ``gene_targets`` property and
    the ``amplification`` / ``knockout`` slices recompute correctly off
    the filtered ``targets``).
    """
    scan = result.scan[
        ~result.scan.index.astype(str).str.startswith(_USAGE_PROT_PREFIX)
    ]
    targets = result.targets[
        ~result.targets["reaction"].str.startswith(_USAGE_PROT_PREFIX)
    ].reset_index(drop=True)
    return FSEOFResult(scan=scan, enforced=result.enforced, targets=targets)
