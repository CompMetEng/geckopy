# kcat aggregation: how far is `max` from `median` on BRENDA?

The BRENDA files (toolbox-assets release `brenda-2026.1-1`) carry **both** the per-(ec, substrate,
organism) max and the per-triple median of all raw measurements that
fell into the triple. They live in the same `kcat.tsv` / `sa.tsv` file
distinguished by an `aggregation` column; `load_brenda_data` splits
them into `BrendaData.kcat_max` / `.kcat_median` (and `.sa_*`) so
consumers never see doubled-up rows. The default
(`kcat_aggregate_brenda = "max"` on `ModelParameters`) preserves
MATLAB-GECKO behaviour; flipping that field to `"median"` (or passing
`fuzzy_kcat_matching(aggregate="median")` per call) makes both
aggregation layers — the per-triple snapshot view and the per-EC
runtime collapse — use median together.

This document quantifies the two layers separately by walking the raw
BRENDA JSON (BRENDA release 2026.1, ~57 900 raw kcat rows, ~39 000
unique triples after dropping trypsin 3.4.21.4 and the
diffusion-limited filter in `parse_brenda_json`):

- [`scripts/kcat_aggregation_raw.py`](../scripts/kcat_aggregation_raw.py)
  walks the raw JSON and answers both questions below.
- [`scripts/kcat_aggregation_compare.py`](../scripts/kcat_aggregation_compare.py)
  reads only the BRENDA TSV (no 710 MB JSON needed) and compares
  per-EC max-of-max vs median-of-median over the pre-aggregated views.

> **3.4.21.4 (trypsin) is excluded** from these analyses: its BRENDA
> entries are known to be unreliable (many non-physiological mutants
> and unrelated substrates).

## Layer 1 — within-triple aggregation

When several raw measurements share an (ec, substrate, organism)
triple, the snapshot has to collapse them. The historical `max`-only
snapshot threw the rest away.

24.5 % of triples (9 549 of 39 053) have ≥ 2 raw measurements.
Across those:

| | quartiles | mean | p90 |
|---|---:|---:|---:|
| within-triple `log10(max / median)` | 0.05 / 0.16 / 0.28 | **0.26** | 0.58 |

So within a triple the max sits typically ~1.8× above the median;
4.8 % of multi-measurement triples have max ≥ 10× the median, 1.2 %
have max ≥ 100×. Small effect — *within* a triple BRENDA measurements
are usually close enough that picking max vs median doesn't move the
needle much.

## Layer 2 — per-EC aggregation

For each EC code with ≥ 5 triples (1 722 EC codes), we collapse
triples to a single per-EC kcat. There are four possible
"snapshot × runtime" combinations:

| snapshot view | runtime aggregate | shorthand | meaning |
|---|---|---|---|
| max | max | **maxmax** | historical default (MATLAB GECKO) |
| max | median | **maxmed** | the runtime-only median flag we had before |
| median | median | **medmed** | adapter `kcat_aggregate_brenda="median"` |
| median | max | medmax | unused; included for symmetry |

`log10` spreads:

| | quartiles | mean | p90 |
|---|---:|---:|---:|
| `maxmax / medmed` (true historical bias) | 0.68 / 1.20 / 1.97 | **1.45** | 2.81 |
| `maxmed / medmed` (snapshot-only swap) | 0.00 / 0.00 / 0.07 | 0.06 | 0.20 |
| `maxmax / maxmed` (runtime-only swap) | 0.63 / 1.14 / 1.90 | 1.38 | 2.77 |

Fraction of ECs where the historical default is N× above the "true"
median-of-medians:

| | fraction of ECs |
|---|---:|
| `maxmax ≥ 10× medmed` | **59.5 %** |
| `maxmax ≥ 100× medmed` | 24.2 % |
| `maxmax ≥ 1 000× medmed` | 8.6 % |

For 60 % of well-measured ECs, the shipped default kcat is ≥ 10× the
typical BRENDA assay; for 8.6 % it is ≥ 1 000× above. Most of that
comes from the runtime aggregation layer (`maxmax / maxmed` ≈ 1.38),
with a small further bias from the snapshot layer (`maxmed / medmed`
≈ 0.06). Both swap together when the user sets
`kcat_aggregate_brenda = "median"`.

| EC | n | max [1/s] | median [1/s] | max / median |
|---:|---:|---:|---:|---:|
| 3.5.2.6 | 601 | 7 150 | 20 | 358× |
| 3.1.1.1 | 355 | 431 000 | 29 | **14 862×** |
| 1.1.1.1 | 318 | 3 500 | 3.1 | 1 129× |
| 3.2.1.21 | 289 | 1 214 290 | 14 | **86 735×** |
| 1.10.3.2 | 269 | 158 300 | 42.1 | 3 760× |
| 3.4.21.4 | 265 | 910 000 | 5.1 | **178 431×** |
| 2.3.2.5 | 264 | 220 | 22.95 | 9.6× |
| 3.1.1.73 | 235 | 14 793 | 10 | 1 479× |
| 3.4.21.83 | 213 | 11 300 | 18 | 628× |
| 1.1.99.18 | 203 | 156 | 17 | 9.2× |
| 1.14.18.1 | 181 | 219 000 | 26.7 | 8 202× |
| 4.1.1.39 | 168 | 23.1 | 4.2 | 5.5× |
| 3.4.21.35 | 165 | 1 380 | 0.81 | 1 704× |
| 2.5.1.18 | 161 | 9 600 | 9 | 1 067× |
| 3.4.24.15 | 156 | 179 | 4.75 | 38× |
| 3.2.1.20 | 139 | 1 278 | 26.2 | 49× |
| 3.1.8.1 | 133 | 30 900 | 11.72 | 2 637× |
| 3.1.2.20 | 128 | 190 | 2.15 | 88× |
| 3.4.24.16 | 122 | 11.9 | 1.55 | 7.7× |
| 3.4.11.1 | 112 | 34 500 | 2.94 | **11 735×** |

| EC | n_triples | maxmax (default) | maxmed | medmed | maxmax / medmed |
|---:|---:|---:|---:|---:|---:|
| 3.5.2.6 | 601 | 7 150 | 20.0 | 15.0 | 477× |
| 3.1.1.1 (esterase) | 355 | 431 000 | 29.0 | **24.0** | **17 958×** |
| 1.1.1.1 (alcohol DH) | 318 | 3 500 | 3.1 | 2.83 | 1 237× |
| 3.2.1.21 | 289 | 1 214 290 | 14.0 | 13.9 | **87 359×** |
| 1.10.3.2 | 269 | 158 300 | 42.1 | 38.0 | 3 760× |
| 2.3.2.5 | 264 | 220 | 22.95 | 22.0 | 10× |
| 3.1.1.73 | 235 | 14 793 | 10.0 | 8.6 | 1 720× |
| 3.4.21.83 | 213 | 11 300 | 18.0 | 17.5 | 646× |
| 1.1.99.18 | 203 | 156 | 17.0 | 13.9 | 11× |
| 1.14.18.1 | 181 | 219 000 | 26.7 | 26.7 | 8 202× |
| 4.1.1.39 (RuBisCO) | 168 | 23.1 | 4.2 | 3.1 | 8× |
| 3.4.21.35 | 165 | 1 380 | 0.81 | 0.81 | 1 704× |
| 2.5.1.18 | 161 | 9 600 | 9.0 | 6.74 | 1 424× |
| 3.4.24.15 | 156 | 179 | 4.75 | 4.0 | 45× |
| 3.2.1.20 | 139 | 1 278 | 26.2 | 25.48 | 50× |
| 3.1.8.1 | 133 | 30 900 | 11.72 | 8.52 | 3 627× |
| 3.1.2.20 | 128 | 190 | 2.15 | 1.68 | 113× |
| 3.4.24.16 | 122 | 11.9 | 1.55 | 1.35 | 9× |
| 3.4.11.1 | 112 | 34 500 | 2.94 | 2.44 | **14 139×** |
| 3.2.1.1 (α-amylase) | 112 | 40 000 | 167.25 | 149.0 | 268× |

Esterase 3.1.1.1 stays the clearest worry: 355 triples, true median
24/s, MATLAB-default 431 000/s (~18 000× above).

## File format on disk

```
# BRENDA release 2026.1 generated 2026-05-28 - CC BY 4.0 - kcat in 1/s
ec_code	substrate	organism	kcat_max	kcat_median	n	references
1.1.1.1	(2e)-but-2-en-1-ol	yokenella sp.	101.0	101.0	1	PMID:24509923
1.1.1.1	(r)-1-indanol	sulfolobus acidocaldarius	7.1	7.1	1	PMID:20049620
...
```

One **wide** row per (ec, substrate, organism) triple. `kcat_max` and
`kcat_median` carry the two statistics computed from the raw
measurements that fell into the triple. `n` is the count of raw
measurements aggregated (so `n=1` rows have max == median, as above).
`sa.tsv` has the same shape with `sa_max` / `sa_median` columns.
`mw.tsv` keeps a single-valued shape (MW is a per-protein physical
property; no aggregation question) — header
`ec_code\tsubstrate\torganism\tmw\tn\treferences`.

## Reproduce

```bash
python scripts/kcat_aggregation_raw.py       # raw-JSON, both layers
python scripts/kcat_aggregation_compare.py   # snapshot-only, per-EC
```

Raw-JSON takes ~6 min on the BRENDA 2026.1 dump (limited by 710 MB JSON
load via WSL2 /mnt/c); snapshot-only runs in ~10 s.
