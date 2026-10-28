# Relaxing proteomics constraints: `flexibilize_enz_concs`, `relax_proteomics_greedy` and `relax_enz_concs`

Both functions answer the same question — *the measured proteomics constraints
make the ecModel grow too slowly (or not at all); which enzyme bounds do we
loosen, and by how much, to restore feasible growth?* — but they are **not**
duplicates. They use different selection signals, different relaxation step
sizes, and produce results of very different stringency. This note records the
differences and the recommendation (keep both; do not merge).
`relax_enz_concs`, a third method that solves the same question as one LP, and
the per-complex options for all of them are described at the end.

## Side by side

| aspect | `flexibilize_enz_concs` | `relax_proteomics_greedy` |
|---|---|---|
| Origin | GECKO MATLAB `flexibilizeEnzConcs.m` (faithful port) | legacy geckopy, Carrasco et al. 2023 |
| Which enzyme to relax | largest **control coefficient** (how much each measured enzyme's bound limits growth, from `get_conc_control_coeffs`: the reduced cost of its usage bound, or a finite difference where the solver exposes no duals) | largest **\|shadow price\|** (LP dual of the usage upper bound) |
| Solves per selection | one LP solve plus one check solve per enzyme with a nonzero coefficient (one solve per candidate protein where the solver exposes no duals) | a **single** `slim_optimize` exposes all duals (cheap) |
| Relaxation step | **gradual**: usage `ub ← conc·(1 + fold_change·k)`; the same enzyme can be picked repeatedly and loosened a bit more each time | **all-or-nothing**: usage `ub ← default_upper_bound` (≈ unconstrained) in one shot |
| Tighten-back | yes — a refinement pass minimises the protein pool at `exp_growth` and **drops** enzymes whose true usage is below their original conc (un-relaxes over-relaxed ones) | none — each picked enzyme stays fully unconstrained |
| Result stringency | **tight** (minimal, proteomics-faithful relaxation) | **loose** (tends to over-relax) |
| Pool fallback | yes — relaxes `prot_pool_exchange` when no single enzyme helps | no |
| Scope | measured enzymes only (`~isnan(concs)`) | any currently-constrained enzyme (`ub < default`) |
| Non-convergence | returns a partial `FlexEnzResult` + warning (global iteration cap) | **raises** `RuntimeError` |
| Result type | `FlexEnzResult` (relaxed enzymes, per-enzyme `frequence`, ratios) | `GreedyRelaxResult` (relaxed→orig-conc dict, step trace, `converged`) |

## Interpretation

- **`flexibilize_enz_concs`** is the right default when you want to stay as
  faithful to the proteomics data as possible: it loosens enzymes a little at a
  time and then tightens back everything that didn't actually need loosening,
  so the final model is the *minimal* deviation from the measurements. The cost
  is speed — the control-coefficient computation re-solves the LP for every
  measured protein on every outer iteration.

- **`relax_proteomics_greedy`** is the right choice when you want a fast answer
  and the infeasibility is dominated by one or two enzymes: a single solve
  ranks all enzymes by shadow price, and fully unconstraining the worst offender
  usually moves growth a lot. The cost is that it **over-relaxes** (no
  tighten-back), so the resulting model is less constrained than necessary, and
  it gives up (raises) rather than degrading gracefully.

## Recommendation

Keep both — they are complementary, not redundant. They could share small
helpers (eligible-enzyme selection, result shaping), but their cores differ
enough that merging into one function would obscure the two algorithms. Two
follow-ups would make them more interchangeable:

1. Give `relax_proteomics_greedy` an optional tighten-back / graded step so its
   output is not systematically looser than `flexibilize_enz_concs` (this is
   the over-relaxation point in the modeling-intent review, §3.3).
2. Align the failure convention — currently one returns a `converged=False`
   result and the other raises; pick one so callers don't special-case each.

## `relax_enz_concs`: the least total relaxation, in one LP

Both methods above relax one enzyme at a time: rank, raise the top one,
re-solve, repeat. Which enzymes end up raised then depends on the ranking
signal and on the order in which enzymes become limiting. On a large model
with noisy proteomics, a small change to the ranking (for example, control
coefficient per mg against gain per fold) can change which enzymes are
raised and, downstream, predictions that depend on the remaining caps.

`relax_enz_concs` asks the question once. Every measured cap becomes
`usage <= conc * (1 + x)`, the protein pool may grow by a fold too, and one LP
minimises the sum of the `x` subject to `growth >= exp_growth`:

- `weight="relative"` (default) minimises summed fold increases, so an enzyme
  measured at almost nothing is not relaxed freely because its absolute cost is
  small; `weight="absolute"` minimises the mg added.
- The pool's fold costs `pool_weight` (default 1000) times an enzyme's, so the
  pool grows only where raising enzymes cannot reach the target, the role of
  the pool fallback in `flexibilize_enz_concs`.
- Folds count from the current caps, so a cap raised earlier is never lowered.
- If no relaxation reaches the target, it raises `ValueError`: growth is then
  not limited by protein.

It returns a `FlexEnzResult` like `flexibilize_enz_concs` (`frequence` is 1 for
every entry). Use it when the outcome should not depend on the order of
relaxation, and to compare with the iterative methods: if they disagree on
which enzymes need raising, conclusions that hinge on those enzymes are
fragile.

## Enzyme complexes

Absolute proteomics recovers the subunits of one complex unevenly: subunits
present 1:1 can come out several-fold apart while their replicates agree within
a few percent. Capped one by one, a complex can then carry no more flux than
its least abundant measurement allows, and relaxation methods that raise one
subunit at a time leave its subunits at unrelated levels, or raise a different
subunit in each condition.

- `level_complex_concs` (after `fill_enz_concs`, before `constrain_enz_concs`)
  gives every complex one abundance: the median per copy over its measured
  subunits, scaled back to each subunit by its copies in `ec.rxn_enz_mat`;
  unmeasured subunits are filled. `level_complex_abundances` does the same on
  molar data before it is written into the model.
- `complex_families` defines the complexes. Two enzyme sets are one complex when
  they share at least two subunits and half the larger set (isozyme variants,
  the same complex on several reactions). A protein in three or more complexes,
  in two with equal weight, or that also catalyses a reaction alone is *shared*
  and keeps its own measurement: electron carriers (cytochrome c), thioredoxin,
  or the E3 subunit of the 2-oxo-acid dehydrogenases, which would otherwise
  chain unrelated complexes together. A subunit in two complexes belongs to the
  one with more reactions (core subunits that also appear in an assembly step).
- `flexibilize_enz_concs(per_complex=True)` and `relax_enz_concs(per_complex=True)`
  raise the measured subunits of a complex by one factor, so levelled
  concentrations stay levelled. In `flexibilize_enz_concs` complexes are ranked
  by the growth gained per fold of raising all their caps (the usage reactions'
  reduced costs times the caps), which leaves out the protein pool's price and
  does not depend on which of several simultaneously binding subunits the solver
  puts the dual on.

Levelling relies on the stoichiometry in `ec.rxn_enz_mat`; where
`apply_complex_data` found no exact ComplexPortal match, every subunit counts
as one copy. Per-complex relaxation without levelling raises every subunit by
the fold its worst-measured subunit needs, which inflates the well-measured
ones; use the two together.
