# /cases — the scheduler benchmark suite

Twenty-eight DAG workloads, written against the [`/vmath`](../vmath/README.md) builder, that
exercise the MPI engine's scheduler across graph sizes, tensor sizes, dtypes, and dependency
shapes. The suite is built to be **fair but unforgiving**: it contains control cases where
naive policies (round-robin, locality-greedy, FIFO) are *supposed* to win, and adversarial
cases constructed so that each naive policy loses for a reason the case's docstring names.
A scheduler cannot get lucky across the whole suite.

Every case is a self-contained script. Submit one directly — `hpcctl` compiles it for you:

```bash
uv run hpcctl submit cases/wide_skewed_001.py -n 4
```

or compile it standalone: `uv run python cases/wide_skewed_001.py > dag.json`. Each script's
docstring states what it measures, why naive schedulers fail (or are expected to win), and
the suggested `-n` sweep. The `dag_id` always matches the filename (underscores → hyphens),
so traces, Slurm logs, and `results/<job id>/` artifacts correlate by name.

## Catalog

### Controls and micro-benchmarks (naive schedulers should tie or win here)

| Case | Graph | Tensors | Measures |
| ---- | ----- | ------- | -------- |
| `tiny_allops_001` | ~14 nodes | (8,8), (3,) | smoke test: all 7 primitives, mixed dtypes |
| `micro_chain_001` | 13 nodes | (16,) | serial chain: per-node + MPI overhead floor |
| `micro_fanout_001` | ~48 nodes | (64,) | fan-out placement, small-message reduction |
| `wide_uniform_001` | ~160 nodes | (128,128) | uniform independent chains — round-robin's best case |
| `montecarlo_paths_001` | ~290 nodes | (8192,) | 24 identical paths: the fairness yardstick |
| `straggler_tail_001` | ~51 nodes | (512,512) | 17 equal tasks: makespan bound is exactly ceil(17/n) |

### Adversarial traps (each one defeats a specific naive policy)

| Case | Defeats | How |
| ---- | ------- | --- |
| `wide_skewed_001` | cost-blind balancing | independent chains spanning ~4000x in cost |
| `trap_critical_path_001` | greedy ready-first | expensive 8-deep spine under 40 cheap distractors |
| `trap_locality_001` | locality-greedy | 8x8 all-to-all shuffle; no placement is local |
| `trap_roundrobin_001` | index-mod-N assignment | heavy/light interleaved so parity classes collide |
| `imbalanced_diamond_001` | FIFO / no upward-rank | diamond arms differing 12x in depth |
| `butterfly_shuffle_001` | static clustering | FFT butterflies re-pair lanes every stage |
| `dtype_cost_skew_001` | shape-only cost models | identical shapes, half float32 half float64 |
| `nbody_lite_001` | eager distribution | hundreds of 24-byte ops; any transfer is a loss |

### Real-world shapes

| Case | Analogue | Structure |
| ---- | -------- | --------- |
| `mlp_forward_001` | model inference | sequential layers with wide activation subgraphs |
| `mlp_ensemble_001` | deep ensembles / serving | 6 heavy pipelines off one shared input |
| `poly_features_001` | feature engineering | shared source, geometrically growing branches |
| `fourier_approx_001` | signal processing | 8 medium harmonic subgraphs + weighted reduce |
| `markov_power_001` | Markov steady-state | strictly sequential matpow chains (scaling floor) |
| `pagerank_lite_001` | PageRank / iterative solvers | 12 iterations against one hot 32 MiB matrix |
| `powmod_residue_001` | modular exponentiation | 12 unsplittable serial powmod lanes |
| `fork_join_ladder_001` | BSP / training steps | 6 fork-join stages; stragglers compound |
| `reduction_tree_001` | distributed aggregation | 64-leaf tournament, width halves per level |
| `wavefront_grid_001` | DP alignment / LU panels | 10x10 lattice, parallelism ramps 1..10..1 |

### Scale extremes (compare runtimes across `-n` and against each other)

| Case | Corner | Shape |
| ---- | ------ | ----- |
| `xl_dense_chain_001` | few nodes, huge matrices | 9 nodes, (2048,2048), 32 MiB operands |
| `xl_vector_stream_001` | few nodes, huge vectors | 8 bandwidth-bound 16 MiB streams |
| `xs_swarm_001` | huge graph, tiny data | ~1300 nodes of (4,) vectors |
| `grand_mixed_001` | everything at once | spine + cloud + wavefront + late series, ~350 nodes |

## Reading the results

- **Ideal-bound cases** (`straggler_tail_001`, `wide_uniform_001`, `montecarlo_paths_001`)
  have computable optimal makespans — report measured/ideal ratios, not raw seconds.
- **Floor cases** (`micro_chain_001`, `markov_power_001`, `xl_dense_chain_001`) must be flat
  across `-n`; any slope is scheduler-inflicted overhead.
- **Trap cases** are diagnosed by *contrast* with their control twin: `wide_skewed_001` vs
  `wide_uniform_001` isolates cost-awareness; `trap_roundrobin_001` at `-n 2` vs `-n 3`
  isolates index-based assignment; `dtype_cost_skew_001` isolates dtype-blind cost models.
- Every case is bit-deterministic (fixed seeds), so runtime differences between runs of the
  same case at the same `-n` are noise + scheduling, never workload variance.

## Conformance

`vmath/tests/test_cases.py` compiles every script exactly the way `hpcctl submit` does and
validates the output against [`/shared/dag_schema.json`](../shared/dag_schema.json); it also
pins the suite at ≥ 20 cases and the `dag_id`-matches-filename convention.
