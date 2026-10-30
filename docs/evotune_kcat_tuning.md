# Tuning kcats against experimental data

A first, brief protocol for fitting an ecModel's kcats to measured
growth rates and/or fluxes with `cmaes_kcat_tuning`. This covers one
recommended path end to end; it is not a full reference (see the
docstrings in `geckopy.kcat_tuning.evotune` for every
option) and it does not build an ecModel from scratch (see
[migrating_from_gecko_matlab.md](migrating_from_gecko_matlab.md) for
that).

## What this does

An ecModel's kcats mostly come from databases (BRENDA), predictors
(DLKcat, OpenKineticsPredictor), or manual curation, and they don't
reproduce measured growth on their own. Three functions, used in
sequence, turn measured data into corrections:

1. **`screen_kcat_leverage`** reports which kcats the data can
   actually speak to -- no optimisation, just "if this kcat were
   different, how much would the fit change." Useful for curation on
   its own, before any tuning run.
2. **`select_tunable_mask`** turns that report into the set of kcats a
   tuning run should search over, automatically if you skip it.
3. **`cmaes_kcat_tuning`** searches that set with CMA-ES for the kcat
   vector that best matches the data, weighted by how much each kcat's
   source is trusted.

A fourth function, **`review_assignment`**, is complementary rather
than part of that sequence: it checks the kcats *themselves* for
assignment mistakes -- independent of any measured data -- since the
tuner's largest corrections are usually places the assignment went
wrong rather than discoveries about biology (see Step 4).

The result is usually a short list of corrections, each with a
before/after value and a source -- not a wholesale rewrite of the
model. A run that changes nearly every kcat by a little has not found
anything; a run that changes a handful by a lot, concentrated in the
least-trusted sources, has.

## Before you start

- A working ecModel and `ModelAdapter` project (an existing
  `model_adapter.toml` and `models/*.yml`).
- `pip install geckopy[evotune]` -- the CMA-ES search
  (`cma`) is an optional dependency, not part of the base install.
- Experimental data: at least one of a set of measured exchange fluxes
  per condition, or a set of measured maximum growth rates per carbon
  source. Both together works and is preferred -- flux data pins the
  *shape* of metabolism, growth rate pins its *rate*.

## Step 1: Prepare the experimental data

Three tab-separated files go under `<project>/data/`. All three are
optional -- a missing file is skipped, not an error -- but you need at
least one of the first two.

| file | contents | required |
|------|----------|----------|
| `evotuneFluxData.tsv` | measured exchange fluxes, one row per condition | at least one of these two |
| `evotuneMaxGrowth.tsv` | measured maximum growth rate, one row per condition, one active carbon source per row (at `-1000`, i.e. unconstrained) | |
| `evotuneZeroExch.tsv` | reaction IDs assumed to carry zero flux in every condition | no |

Both flux files share one column layout (the same parser geckopy uses
for regular flux data):

```
Condition   Ptot   grRate   glucose (r_1714)   fructose (r_1709)   ...   evotuneRMSEweight   source
glucose     NaN    0.41     -1000              NaN                ...   1                     DLKcat
fructose    NaN    0.338    NaN                -1000              ...   1                     DLKcat
```

`Condition` names the row, `grRate` is the measured growth rate,
`Ptot` is measured protein content (`NaN` if not measured), each
`<met> (<rxn>)` column is a measured or fixed flux for that exchange
reaction, `evotuneRMSEweight` scales how much that row counts toward
the RMSE (usually `1`), and `source` is free text carried through for
your own bookkeeping. `evotuneZeroExch.tsv` is simpler -- a `Rxns`
header, then one reaction ID per line.

## Step 2: Configure `[evotune]` in `model_adapter.toml`

Every field has a default, so this section can be omitted entirely to
start. A configured example, in plain terms:

```toml
[evotune]
sigma0_log_default = 0.3   # trust for any source not listed below
max_growth_weight = 2.0    # weight the growth-rate data double against flux data

[evotune.source_groups.okp]
sources = ["OpenKineticsPredictor"]
match_okp = true

[evotune.source_groups.brenda]
sources = ["brenda"]

[evotune.source_groups.custom]
sources = ["custom"]

[evotune.sigma0_log_source]
okp = 0.25
brenda = 0.2
custom = 0.1
```

What each knob means:

- **Trust tiers** (`source_groups` + `sigma0_log_source` +
  `sigma0_log_default`). Group your kcats by how much you trust their
  origin, and give each group a standard deviation in log-space:
  smaller means more trusted, and the search needs stronger evidence
  before it moves that kcat far. `custom` (curated by hand) at 0.1 is
  three times more trusted than an unlabelled prediction at 0.3. This
  is the one section with no universal default -- your model's own
  source labels decide it.
- **`max_growth_weight`** (default `1.0`). The combined score is
  `(rmse_flux + w * rmse_max_growth) / (w + 1)`. At `2`, growth-rate
  error counts double against flux error -- useful when growth rate is
  measured across many carbon sources but flux is mostly one.
- **`prior_penalty_weight`** (default `0.03`, already on). Adds
  `w * mean(((log k - log k0) / sigma0)^2)` to the score: a charge for
  moving a kcat, scaled by how much you trust its prior. Without this
  the fit is often flat along many directions -- several very
  different kcat vectors fit equally well -- and the search returns an
  arbitrary point along that flat direction, which does not reproduce
  if you rerun it with a different seed. The default is calibrated so
  that large corrections agree in direction and land within a handful
  of fold of each other across seeds, at a small cost in raw fit. Set
  it to `0` to score on fit alone (also what a run reproducing GECKO
  MATLAB, which has no such term, must do). The right strength is
  model- and data-specific, since it depends on the scale of
  `sigma0_log` and of the objective itself --
  `tune_prior_penalty_weight` runs a candidate sweep and reports fit
  cost against cross-seed reproducibility so it doesn't have to be
  picked blind (see its docstring; it costs one full tuning run per
  candidate per seed, so treat it as an occasional calibration step,
  not part of routine tuning).
- **`tie_isozymes`** (default `true`). Several ecModel reactions are
  one enzymatic reaction split across isozyme copies. When copies
  share a prior value and a source, no experimental condition can tell
  them apart, so an untied search is free to invent a difference
  between them that means nothing biologically. Tying gives such
  copies one shared kcat.
- **`max_generations` / `rmse_threshold`**. Double as this search's
  stopping conditions: run at most this many CMA-ES generations, or
  stop earlier once the best RMSE reaches `rmse_threshold` (negative
  never stops early).

## Step 3: Find out what the data can see

```python
from geckopy import ModelAdapter, load_ec_model
from geckopy.kcat_tuning.evotune import screen_kcat_leverage

adapter = ModelAdapter.from_folder("path/to/project")
model = load_ec_model(adapter=adapter)  # models/ecModel.yml by default

screen = screen_kcat_leverage(model, adapter=adapter, n_proc=8)
screen.drop(columns="_positions").head(20)
```

This perturbs every tunable kcat (isozyme copies moved together, per
`tie_isozymes`) up and down and measures how much the fit changes --
no tuning happens yet. The result is a table, one row per kcat or tied
group, ranked by that leverage weighted by trust: the reaction ID, how
many isozyme copies it covers, its source group, its current value,
its leverage, and `cum_leverage_share` -- the running fraction of
total leverage carried by this row and every row above it. The top
rows are what a curator should look at first, independent of whether
you ever run a tuning search: on ecYeastGEM, a handful of kcats
routinely carry the majority of what any tuning run could achieve, and
they tend to be implausible database values rather than genuine
biology.

This costs one simulation per condition per kcat group, roughly
comparable in scale to a full tuning run's budget -- expect tens of
minutes on a genome-scale model, not a quick check. `n_proc`
parallelises it the same way tuning does.

## Step 4: Check the assignments behind the top kcats

Before spending a tuning run on a kcat, it is worth checking whether
the value was assigned correctly in the first place.
`review_assignment` flags source-level problems: a value with no kcat
row behind it for its EC, one number reused across many unrelated
reactions, an EC's maximum standing in for a whole distribution, an
implausible magnitude, or a curated value that just repeats what the
database already said. These are corrections to the *source* -- fix
them in whatever database or fuzzy-matching step in `gather_kcats`
produced the value -- so they propagate to every model built from the
same data, unlike a value the tuner moves for this one model alone.

```python
from geckopy.gather_kcats.review import EcStats, review_assignment, findings_tsv

# One EcStats per EC code queried while gathering kcats -- summarise
# whatever kcat database (usually BRENDA) the assignment came from.
ec_stats = {
    "1.1.1.1": EcStats(n_kcat=12, kcat_max=340.0, kcat_median=45.0,
                       values=frozenset({340.0, 45.0, 12.0}), n_sa=0),
    # ...
}

findings = review_assignment(
    model.ec.rxns, model.ec.kcat, model.ec.source, model.ec.eccodes,
    ec_stats=ec_stats,
)
print(findings_tsv(findings))
```

Each `Finding` names a reaction, its checks (`no-ec-evidence`,
`repeated-value`, `ec-maximum`, `magnitude`, `custom-duplicate`), and a
human-readable `detail`. Isozyme copies the
assignment could not tell apart (the same grouping `tie_isozymes` in
Step 2 uses) are reported once, not once per copy. Standard fallbacks
and tied groups are deliberate choices, not mistakes, so both stay
quiet unless asked for via `include`.

Findings rank by `leverage` when you pass it in -- reuse the `screen`
from Step 3, expanded from tie-groups back to individual
`model.ec.rxns` positions via its `_positions` column -- so that
looking odd and actually mattering are not confused; without it, rows
come back in input order, and looking odd is all you get. `coverage`
(a share of flagged leverage) or `top` (a row count) then truncates the
report: on ecYeastGEM, tightening the checks themselves does not work
half as well as this does, since `coverage=0.8` cuts 2575 unfiltered
rows to 33, and the first row alone carries half.

## Step 5: Tune

```python
from geckopy import save_ec_model
from geckopy.kcat_tuning.evotune import cmaes_kcat_tuning

result = cmaes_kcat_tuning(
    model, adapter=adapter, screen=screen, n_proc=8, seed=0,
)

save_ec_model(model, "ecModel_tuned.yml", adapter=adapter)
```

Passing the screen from Step 3 avoids recomputing it; omit `screen`
entirely and `cmaes_kcat_tuning` runs one itself. Either way, the
tunable set is built by `select_tunable_mask`, which keeps the fewest
highest-ranked kcats whose combined leverage reaches
`target_impact_share` (default `0.9`) of the total -- a cutoff
relative to the model's own achievable improvement, not a fixed count
or an absolute leverage value, so the same setting means a comparable
thing on a different model. Pass `tunable_mask` directly instead if
you want to fix the set yourself.

`model` is mutated in place: on return, every tunable kcat carries the
best value CMA-ES found, and the model's kcat constraints are already
applied -- nothing further is needed before simulating or saving it.
Beyond `target_impact_share`, the only other knobs are `popsize`
(CMA-ES's population size, defaulting to its own dimension-scaled
choice rather than a value tuned for one particular model), `n_proc`,
and `seed`; everything else comes from the same `EvotuneParams` as
Step 2, so there is nothing new to configure once you've read it.

If growth needs special handling for your organism -- forcing
anaerobic conditions, or scaling the protein pool with a
condition-specific biomass composition -- pass `make_anaerobic` and/or
`change_protein_biomass` callables; geckopy has no organism-agnostic
default for either. See `tutorials/full_ecModel/code/anaerobic.py` in
this repository for a worked example.

## Step 6: Read the result

`result` is an `EvotuneResult`:

- `rmse_trace` / `objective_trace` -- best-so-far plain-fit RMSE and
  best-so-far optimised objective, per generation (identical unless
  `prior_penalty_weight` is nonzero). A flat trace for many generations
  before the run ends means it stalled, not converged.
- `new_kcat` / `old_kcat` / `rxns` / `groups` -- the tuned and prior
  kcats over the searched set, parallel arrays, with each kcat's trust
  tier. Tied isozyme copies share one value in `new_kcat`.
- `converged` -- whether `rmse_threshold` was reached, or
  `max_generations` was hit first.
- `n_generations` -- how many generations actually ran.

Before trusting a tuned kcat, check more than the final RMSE:

1. **How many kcats changed**, and by how much -- a result that moves
   nearly everything by a little has not identified anything.
   `geckopy.kcat_tuning.evotune.parsimony` has the
   tools for this: `n_changed`, `fold_change`, `source_movement`.
2. **Impact share** -- `parsimony.impact_share` reports what fraction
   of the total achievable improvement the changed kcats actually
   carry, distinguishing "few, large, and consequential" from mere
   sparsity. This is a different, unweighted quantity from
   `screen_kcat_leverage`'s trust-weighted `cum_leverage_share`.
3. **Reproducibility.** Rerun with a different `seed`. Large
   corrections (say, beyond two-fold) should land in the same
   direction and within a small factor of each other; if they don't,
   the search has found a flat direction rather than a real
   correction, and `prior_penalty_weight` is the first knob to reach
   for.
4. **Whether it generalises**, if you have data you didn't fit on --
   score the tuned model against it and compare to the untuned model.
   A large improvement on fitted conditions with no improvement
   elsewhere is a sign of overfitting to a thin dataset, not a fixed
   model.

`new_kcat`/`old_kcat` is thousands of numbers, most of them unchanged --
not something a reviewer can read. `result.corrections()` turns it into
the actual review artifact: one `Correction` row per changed kcat
(reaction, EC code, source, prior/tuned value, fold change), ranked by
*leverage* rather than by how far a kcat moved, so the parameters the
data cannot see (free to drift furthest, least worth trusting) sort to
the bottom instead of the top:

```python
from geckopy.kcat_tuning import annotate_from_model, corrections_tsv

# screen is Step 3's table -- reuse it rather than recomputing leverage.
leverage_by_rxn = dict(zip(screen["rxn_id"], screen["leverage"]))
leverage = [leverage_by_rxn.get(r, 0.0) for r in result.rxns]

annotations = annotate_from_model(model, result.rxns)
rows = result.corrections(leverage=leverage, **annotations)
open("corrections.tsv", "w").write(corrections_tsv(rows))
```

`cumulative_share` on each row is the running fraction of total leverage
the list has accounted for so far, so a reader can stop where it stops
climbing rather than reading every row. This is the tuning-time
counterpart to `review_assignment` in Step 4: that flags kcats worth
curating *before* a run; this reports which ones the run actually
changed, and how much each change is worth trusting.
