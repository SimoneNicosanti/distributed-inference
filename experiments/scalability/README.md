# Scalability benchmark

Synthetic, nested optimization instances. Builds the optimizer's normalized
`CompactOptimizationProblem` directly and uses its full (default) or incremental strategy, MISOCP
formulation, Gurobi compiler, solver and translator. Static profiling still uses
the model manager's complete `DefaultModelProfiler` and controller profile types.
No microservices, Redis, HTTP endpoints, worker processes or GPU hardware are
needed to construct the simulated infrastructure. Real Gurobi execution needs an
adequate license.

The compact MISOCP supports two layer representations: `assignment` (default)
and `cumulative`. The old `compact` name remains an alias for `assignment`.
Select with `solver.formulation` in YAML or `--formulation` on the CLI; `legacy`
is rejected.
The benchmark exposes five optimization objectives, including
`arrival_weighted_normalized_response_time`. The optimizer's `CONSTANT`
feasibility objective is excluded from configuration and `--objectives all`.
Profiles must contain `max_cut_size_mib`. Historical result files remain readable.

## Separate experiments

Each experiment has its own command, configuration and default output:

| Experiment | Command | YAML | Output |
| --- | --- | --- | --- |
| Performance requirements; optimizer chooses models | `scalability-performance` | `config/performance.yaml` | `results/performance` |
| Every flow explicitly requests EfficientNet B6 | `scalability-model-variant` | `config/model-variant.yaml` | `results/model-variant` |

```bash
# Inspect independent plans without profiling or solving.
uv run --project experiments/scalability --locked scalability-performance plan
uv run --project experiments/scalability --locked scalability-model-variant plan

# Run either experiment independently.
uv run --project experiments/scalability --locked scalability-performance main
uv run --project experiments/scalability --locked scalability-model-variant main
```

`scalability` remains the performance command. The campaign CLI no longer accepts
`--model-selection` or executes both modes together. Each command rejects a YAML
for the other experiment. Both retain `pilot`, `main`, `sensitivity`, `plan`,
`analyze`, `--config`, `--output`, `--strategy`, `--formulation` and
`--objectives`.

The performance YAML retains all ten models and the `main.models`,
`sensitivity.models` and pilot `models` fields. Its baseline is restored from revision `611dc9e`: flows 1–7, model counts 1–10,
ResNet-50 calibration, deadlines 0.2–3 seconds, up to seven components and three
priorities. The independent experiment mode and output directory are retained.

The model-variant YAML contains only `efficientnet_b6`. Its main, sensitivity and
pilot sections omit `models`: the schema supplies exactly one candidate and
rejects a candidate-count sweep. Every request has `model_variant_id` set to B6,
no performance threshold, and exactly one eligible variant. Missing B6 is an error.
The default calibration reference is B6 on the device. With
`arrival_weighted_accuracy`, model choices are fixed and the objective is zero.

The instance generator, compiler, solver, persistence and analysis remain shared.
Run identities, JSONL, CSV and aggregates retain `model_selection`; JSONL flow
records contain the assigned `model_variant_id`. Historical records without a
selection mode are interpreted as `performance`. Default outputs, manifests and
profile caches are independent. An explicit `--output` overrides that destination.

The paired `scripts/compare_formulations.py` helper keeps its single-mode
`--model-selection model_variant` option and automatically selects the B6 YAML
unless `--config` is supplied. Both formulations receive identical requests.

## Assignment versus cumulative

```bash
uv run --python 3.14 --project experiments/scalability --locked scalability main --formulation assignment --output experiments/scalability/results/main-assignment
uv run --python 3.14 --project experiments/scalability --locked scalability main --formulation cumulative --output experiments/scalability/results/main-cumulative

# Repeated paired runs: one generated scenario, identical settings, alternating order.
uv run --python 3.14 --project experiments/scalability --locked python experiments/scalability/scripts/compare_formulations.py --limit 1 --repetitions 3 --output experiments/scalability/results/paired-formulations
```

Each paired case uses a single generated instance for both representations.
`--limit` counts base instances, so the command above performs six native solves.
Run IDs, CSV columns and metadata include the formulation. Analysis separates
plots into `analysis/assignment/` and `analysis/cumulative/`; aggregates never
pool the two. Solver metrics include original variables/binaries/constraints,
linear and quadratic-constraint nonzeros, presolve removals, node count,
first-incumbent time, a sampled bound trace and zero-tardiness discovery time
where applicable. Full presolved matrix sizes/nonzeros and root-relaxation time
are not synthesized from these observations; enable solver logging for those.
See [mathematics, validation and measured comparison](../../docs/optimization/cumulative-formulation.md).

## Incremental strategy

Select `--strategy incremental` or `solver.strategy: incremental` in YAML.
The optimizer solves flow 1, then flows 1–2, then flows 1–3, and so on in input
order. Every prefix is unrestricted. The previous solution supplies only a
partial MIP start: earlier models, partitions, placements and priorities can
all change. With the B6 experiment, model choice remains fixed by the requests.
Both `assignment` and `cumulative` are supported; `full` remains the default.

Each prefix optimizes to the configured gap or timeout, without a first-solution
limit. A timeout with a validated incumbent can continue; a prefix with no valid
incumbent stops the strategy. There is no separate repair or final solve: the
last prefix already optimizes the complete problem. Thus F flows require F solves.

Optional YAML `insertion_time_limit_s` overrides the ordinary timeout for every
prefix. The budget is F times this per-solve limit, excluding construction.
The obsolete repair/final timeout fields are removed. JSONL contains prefix
diagnostics; CSV includes `successful_insertions` and `native_solve_count`.
The final gap and bound belong to the complete last prefix; discovery histories
remain per solve. Analysis keeps full and incremental strategies separate.
See [algorithm, configuration and validation](../../docs/optimization/incremental-flow-strategy.md).

## Persistent incremental strategy

Select `--strategy incremental-persistent` or
`solver.strategy: incremental-persistent`. It uses the same unrestricted
flow-by-flow optimization and partial MIP starts as `incremental`, while keeping
one native Gurobi model alive. It adds new-flow blocks, updates shared network
and memory rows, and refreshes compute-queue constraints and bounds. Earlier
flow decisions remain free. Both assignment representations and experiment
modes are supported.

```bash
uv run --project experiments/scalability --locked scalability-model-variant main \
  --strategy incremental-persistent --formulation cumulative \
  --output experiments/scalability/results/model-variant/incremental-persistent
```

The existing `incremental` option remains available for comparison. Prefix time
limits and the F-solve budget are the same; identities, diagnostics and analysis
directories distinguish the strategies. Wall-time boxplots include construction
and update costs. No large-model runtime improvement has yet been measured.
See [persistent compiler, bounds and validation](../../docs/optimization/persistent-incremental-flow-strategy.md).

## Setup

Run commands from the repository root. Python 3.14 or later is required.

`uv` installs the optimizer from `services/optimizer` as a local editable
dependency of the benchmark.

The benchmark retains the controller dependency for raw profile types and their
DTO mapper; it does not use controller optimization requests, context, factory or
optimizer transport mappers. Each variant has exactly one named input-shape
configuration. CPU/GPU execution profiles and RAM/VRAM capacities retain their
existing meanings. Optimizer source changes participate in provenance hashes.

```bash
uv sync --project experiments/scalability --locked
uv run --project experiments/scalability --locked scalability plan
```

The project has its own manifest and lockfile. Production manifests and lockfiles
are unchanged. Plotting uses matplotlib; exporting additionally needs the export
dependency group. Current validation is recorded in `VALIDATION.md`; no environment
sync or dependency installation was performed for the optimizer migration.

The export group resolves `torch` and `torchvision` exclusively from the explicit
PyTorch CPU index. It does not install CUDA, NVIDIA runtime packages or Triton.

## Explicit model preparation

The performance experiment needs all ten ONNX files in
`experiments/scalability/models/`; fixed B6 needs only `efficientnet_b6.onnx`.
Prepare the performance catalog explicitly:

```bash
uv sync --project experiments/scalability --locked --group export
uv run --project experiments/scalability --locked --group export python experiments/scalability/scripts/export_models.py
```

To prepare only the B6 experiment catalog:

```bash
uv run --project experiments/scalability --locked --group export python experiments/scalability/scripts/export_models.py --config experiments/scalability/config/model-variant.yaml
```

Default: deterministic random weights, no downloads. The benchmark uses synthetic
accuracies, so weights do not represent measured predictive quality. To supply
trained weights, put torchvision state dictionaries named `resnet50.pth`, etc.,
in a local directory:

```bash
uv run --project experiments/scalability --locked --group export python experiments/scalability/scripts/export_models.py --weights local --weights-directory /path/to/weights
```

`--weights download` explicitly permits torchvision pretrained-weight downloads.
Neither the runner nor the default exporter downloads model files. Existing model
files cause an error unless `--force` is supplied.

To add only the four new models while retaining existing artifacts:

```bash
uv run --project experiments/scalability --locked --group export python experiments/scalability/scripts/export_models.py --models efficientnet_b1 efficientnet_b3 efficientnet_b5 efficientnet_b6
```

The script uses eager export (`dynamo=False`), FP32, static batch 1, opset 17,
input `images`, output `logits`, and deterministic unique ONNX node names.
It then invokes the existing `OnnxModelOptimizer` at BASIC optimization level.
This keeps standard ONNX operators; extended provider-specific fusions may not be
supported by the existing static profiler. Raw exports remain in `models/source/`.
Each final artifact has an `.export.json` manifest with export settings, package
versions and artifact hashes. Shape, export seed, opset and weight mode are YAML
settings, with explicit command-line overrides. Keep the local weight files and
export manifests when archiving an experiment.

The pool contains ResNet-50/101/152 and EfficientNet-B0 through B6.
Default image sizes: ResNet-50/101/152 and EfficientNet-B0 224x224,
B1 240x240, B2 260x260, B3 300x300, B4 380x380, B5 456x456,
B6 528x528, three channels. The exporter supports `--models NAME [NAME ...]`
to select configured entries without replacing the others.
Dynamic symbols in manually supplied models receive exactly one configured value.
Contradictory static dimensions and anonymous dimensions fail validation.
The existing profiler currently requires input `images`; it is not modified.

## Commands

```bash
# Six representative runs, native solver timeout 60 seconds, then offline analysis.
uv run --project experiments/scalability --locked scalability pilot --output experiments/scalability/results/pilot --analyze

# Performance main: seed 42, load 0.7, 7 flow counts x 10 model counts x 11 infrastructures.
uv run --project experiments/scalability --locked scalability main --output experiments/scalability/results/main

# Resume: completed timeouts and proven infeasibility are completed outcomes.
uv run --project experiments/scalability --locked scalability main --output experiments/scalability/results/main --resume

# Retry previously recorded execution errors explicitly.
uv run --project experiments/scalability --locked scalability main --output experiments/scalability/results/main --resume --retry-errors

# Optional sensitivity, explicitly selected; never part of an automatic main run.
uv run --project experiments/scalability --locked scalability sensitivity --output experiments/scalability/results/sensitivity

# Read persisted results only. Does not profile or invoke the solver.
uv run --project experiments/scalability --locked scalability analyze --output experiments/scalability/results/main
```

All commands accept `--config PATH` and `--output DIRECTORY`. Repeat `--seed` to
override the seed list. Execution commands support `--limit N` for bounded manual
checks. YAML paths resolve relative to `root`, itself relative to the YAML file.
`plan` enumerates configurations without loading models or starting a solver.
Campaign `enabled` flags describe default inclusion; explicitly selecting a named
campaign opts into it, including sensitivity even with its default `enabled: false`.
There is no command that implicitly executes both campaigns.

Current counts per objective:

| Experiment | Pilot | Main | Sensitivity | Main solver budget |
| --- | --- | --- | --- | --- |
| Performance | 6 | 770 | 198 | 64 h 10 min |
| Fixed B6 | 6 | 22 | 66 | 1 h 50 min |

Both main grids use seed 42, load 0.7, eleven infrastructures and a 300-second
solve limit with relative gap 0.01. Performance uses flows 1–7; fixed B6 uses flows
5/7. Sensitivity is optional. Construction,
profiling, persistence and solver shutdown add time; these budgets are not strict
wall-clock bounds. Use each command's `plan` after changing its YAML.

## Objective selection

Each run optimizes exactly one objective. Accepted YAML/CLI values:

| Objective | Behavior |
| --- | --- |
| `arrival_weighted_response_time` | Minimize arrival-weighted response; enforce every response deadline. Default YAML objective. |
| `arrival_weighted_normalized_response_time` | Minimize the arrival-weighted fraction of each deadline consumed; enforce every response deadline. |
| `arrival_weighted_accuracy` | Maximize arrival-weighted accuracy; enforce every response deadline. |
| `deadline_violations` | Minimize the number of missed deadlines using binary indicators. |
| `arrival_weighted_normalized_delay` | Minimize arrival-weighted positive delay divided by each flow's deadline. |

```bash
# Plan all objectives without loading models or running a solver.
uv run --project experiments/scalability --locked scalability plan --objectives all

# Run all five independently on the same generated instances.
uv run --project experiments/scalability --locked scalability main --objectives all --output experiments/scalability/results/main-objectives

# Select only a subset.
uv run --project experiments/scalability --locked scalability main --objectives arrival_weighted_accuracy deadline_violations --output experiments/scalability/results/accuracy-violations
```

Omitting `--objectives` uses `optimization.objective` from YAML. Names are
deduplicated while preserving order; `all` must be used alone. There is no
hierarchical optimization or response-time tie-break. The former response-slack
objective names are replaced by the values above.

The counts/budgets above are **per objective**. Selecting all five gives 3,850
performance main runs or 110 fixed-B6 main runs;
solver time budgets scale by the number selected. `--limit N` limits total runs
after objective expansion. Run IDs include the objective, so `--resume` skips
only completed runs for that objective. Instance generation and profile caches
are shared across the selected objectives. CSVs, aggregates and plots identify
each objective separately; historical records without objective metadata form
an `unspecified` group rather than being pooled with a current objective.
Code/provenance changes still invalidate old run IDs; runs from before the
objective/bound changes are historical comparisons, not equivalent resumed runs.

Accuracy uses the flow's requested metric/dataset (accuracy or mAP); error metrics
MAE/RMSE have no automatic conversion into an accuracy score. Deadlines no longer
clip service-time/queue bounds or response/transmission variables for soft
objectives. Response epigraphs may be conservative when smaller response is not
rewarded by the selected objective. The synthetic benchmark accuracy remains
the YAML-generated score rather than a measured prediction metric.

## Synthetic assumptions

All assumptions are in `config/performance.yaml` and `config/model-variant.yaml`; changes are included in result IDs.
There is no automatic calibration adjustment, deadline relaxation or feasibility
repair. Report pilot findings before changing assumptions.

* Independent addressed random streams: model order, synthetic accuracy, layer/tier
  execution jitter, worker-pair networking, flow arrival weights, and deadlines.
  They use SHA-256-derived seeds, not Python's randomized `hash()`.
* Accuracy increases strictly with the explicit global YAML ranking. This ranking
  is synthetic, not measured ImageNet performance. Candidate sets are prefixes of
  a separate seeded permutation. `PerformanceMetricType.ACCURACY` uses `>=` with
  matching dataset; every flow's minimum is zero.
* One full scenario is generated per seed: ten models, seven flows, eleven workers
  and 110 directed links. Instances select prefixes/subsets without resampling.
  Worker IDs remain stable, and identical tiers share execution profiles.
* Infrastructure: device alone, then one through ten edges. The current campaign
  does not select cloud workers; cloud hardware and link profiles remain available.
* Device: four CPU cores, 8 GiB RAM. Edge: 8 GiB VRAM, 16 GiB physical RAM.
  Cloud: 24 GiB VRAM, 64 GiB physical RAM. Edge/cloud host core counts are explicit
  synthetic assumptions (8/16), not parallel service multipliers.
* **Backend-aware profiles:** device estimates use `Backend.CPU`; edge/cloud
  estimates use `Backend.GPU`, matching the worker resource profiles. Inference
  capacity is device RAM or GPU VRAM (8/24 GiB); host RAM remains separately
  represented as physical RAM. Workers and execution metadata record the effective
  backend. These are synthetic estimates; no GPU hardware is needed to run them.
* Operator time: `overhead + max(FLOPs / effective_compute, traffic / effective_bandwidth)`.
  Traffic estimates input/output tensors and weights once per operator. FLOPs
  come from the existing static profiler. YAML defines operator-group efficiencies,
  tier throughput/bandwidth and overhead. Unlisted operators use the documented
  fallback; their names are recorded. Small operators can run slower on GPUs.
* Fixed layer/tier jitter is uniform in [0.9, 1.1]. It does not represent temporal
  service variance. SCV is zero. The existing queue formulation squares each
  component's summed service time; do not sum layer second moments instead.
* Diagnostic partition memory is unique persistent weights plus peak live tensors
  in the existing topological order. Inputs live until their last consumer;
  outputs crossing the partition remain until completion. No in-place aliasing,
  allocator reuse, concurrency or operator workspace is assumed.
* **Memory constraint remains unchanged:** production sums layer weights per
  deployed component and uses a worker maximum over component-summed activations.
  Diagnostic peak-liveness estimates do not replace these coefficients.
  `LayerExecutionProfile.memory_bytes` is not used by the production memory constraint.
* Network ranges are configurable, synthetic, reciprocal pair samples. Store
  bytes/s and RTT seconds. RTT is twice sampled one-way latency; the formulation
  already adds RTT/2 once. It retains original cut-tensor transmission coefficients
  and source-worker network sharing. No new NIC scheduler or routing is introduced.
* Seven Poisson flows have independently sampled weights in [0.8, 1.2]. Reference
  service time is the complete ResNet-50 CPU model on the device, always available
  for calibration even if absent from a candidate prefix. For load `L`:

  `lambda_i = L * weight_i / (reference_service_time * sum(all_seven_weights))`.

  Main and pilot use `L = 0.7`. Sensitivity retains loads 0.3 and 0.8.

  Thus all seven reference flows together offer utilization `L`. Smaller prefixes
  do not rescale rates. Sensitivity scales arrivals only: IDs, deadlines, models,
  execution, workers and networking remain identical for a seed.
  Expanding the old six-model/five-flow pool changes model permutation, synthetic
  accuracies and arrival-rate calibration relative to that earlier configuration.
  This defines a new scenario with new IDs; do not treat old results as completed
  runs of the expanded campaign. Prefixes within the new scenario remain stable.
* Deadlines are uniform in [0.2, 3] seconds for all campaigns. Ingress and sink
  are fixed to the device, which is also an inference runner. The required finite energy limit is
  explicit YAML metadata; the current formulation does not enforce energy.
* Seven components use the existing formulation's component definition, including
  its input/output boundary layers. Priorities, queue discretization, objective,
  utilization constraints and transmission accounting are reused unchanged.

## Solver evidence and reproducibility

The benchmark composes the production `FullSolverStrategy`: compact data builder,
Gurobi model builder/compiler, solver-config builder, native solver and compact
translator. Formulation configuration, including the objective, is bound to the
strategy constructor; `solve` receives only requests and context.

A recording solver retains `SolverOutput` before translation and model closure.
Normalized termination, optional solved model, objective, bound and runtime come
from this output; `GurobiRunStatistics` contains native diagnostics and incumbent
history. A missing incumbent raises `SolveError` in the full strategy, while the
recorded output still distinguishes timeout and proven infeasibility. Native
exceptions remain solver errors; no unavailable measurements are invented.
Explicit benchmark settings: TimeLimit 300 (pilot 60), MIPGap 0.01, Seed 0.
Threads remains unset by default, preserving project behavior; effective
`Threads=0` means Gurobi automatic selection. Host CPU count/affinity and package
versions are recorded. Record native status alongside the benchmark label:

* `optimal_closed_gap`: native OPTIMAL and reported gap zero, within numerical tolerances.
* `gap_target_reached`: native OPTIMAL and a positive observed gap within target.
* `optimal_within_tolerance`: native OPTIMAL with another/unknown gap, e.g. another tolerance.
* `timeout_with_incumbent`, `timeout_without_incumbent`.
* `proven_infeasible`: native INFEASIBLE only; INF_OR_UNBD remains separate.
* `other_termination`, `interrupted`, `solver_error`.

The positive-gap label describes observed evidence; it does not infer an
unreported internal stopping cause. Missing objectives/gaps are null, not zero.
Infinite bounds/gaps are unavailable. Wall time, native runtime and construction
time are distinct. Construction measures native environment/model construction;
compact data preparation is included in wall time. The benchmark enables
`GurobiSolverConfig.record_incumbent_history` (default false for
other callers). `time_to_first_incumbent_s` records the first finite MIPSOL
candidate using native solver runtime, excluding construction. It appears in
JSONL and `summary.csv`. Strict improvements are filtered for minimization or
maximization; non-improving candidates are ignored.

For incremental strategies, `construction_time_s` sums prefix preparation and
native construction/update time. `prefix_wall_time_s` sums complete prefix
attempts, including solves. It replaces the misleading name
`incremental_construction_time_s`; CSV export also recognizes that historical
field without changing archived JSONL records. Model dimensions use one shared
collector across scalability, bottleneck and priority experiments.

`raw_solver_statistics.incumbent_history` in each finished JSONL event contains
`{runtime_s, objective_value}` samples at 0, 1, 2, ... seconds, plus termination
when it falls between whole seconds. Samples carry forward the best objective
observed by that time; before the first incumbent the value is null. A zero
objective remains zero. The trace is reconstructed from timestamped improvements
because callbacks are not invoked on a fixed timer. Multiple improvements within
a second contribute the best value available at its next sampling boundary.
The native final objective is included at termination. If no discovery callback
was observed, time-to-first remains null rather than using termination time.

These observations are buffered in memory and persisted with the finished run,
not streamed during optimization. A process crash loses the unfinished trace;
resume reruns that attempt. Existing completed runs are not backfilled.
No additional bound/gap progress trace is collected.

Source hashes include dirty production files, the experiment and its lockfile.
Artifact hashes include ONNX external-data sidecars. Cached static profiles are
keyed by artifacts, shape, accuracy, source and dependency versions. Run IDs also
include effective scenario configuration, campaign, timeout and software/host
identity. Changing any of these starts a new set of runs rather than reusing
incompatible results. Fixed instances are reproducible; wall-clock-limited solver
results and runtimes need not be identical across repetitions.

References: [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html),
[Gurobi parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html),
[Gurobi callbacks](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/callbacks.html).

## Persistence and analysis

`runs.jsonl` is the authoritative append-only journal. Each attempt writes a
started event before solving and a finished event afterward, with fsync.
`summary.csv` is an atomically regenerated latest-attempt view. Full immutable
scenario/software manifests are in `scenarios/<scenario_id>.json`; run JSONL
contains scenario references, selected model identities, flow metadata and solver
configuration. Retain the entire output directory, model artifacts and sources.
The first started event for each processed seed also embeds full scenario/software
metadata in JSONL; subsequent run events reference its scenario ID.

A process lock prevents concurrent writers to one output directory (POSIX).
Resume discards only a trailing line without a commit newline. Corruption in a
committed line is an error. Incomplete/interrupted attempts are rerun; finished
timeouts and proven infeasibility are not. Solver errors need `--retry-errors`.
Ctrl-C/process termination leaves an incomplete event safe for resume. Because
the production solver runs in a thread, Ctrl-C shutdown may wait for its current
native solve to return; the incomplete attempt is rerun on resume.

Offline analysis writes `analysis/runs.csv`, `aggregated.csv` and PDF/PNG figures
for workers, flows and candidate models. Individual runtime points distinguish
timeouts. Aggregation includes timed-out runtimes and reports observed sample
counts; missing/error runtimes are not invented. Discovery/timeout/infeasibility/
error rates use all completed attempts as denominator. Gap summaries use only
available incumbent gaps. Other scaling factors and seeds are pooled explicitly;
raw CSV allows conditional analysis. With one seed, these are descriptive plots,
not estimates with independent-replication confidence intervals.

The same `analyze` command also generates these PDF/PNG files when the required
observations are present:

* `analysis/<campaign>-<objective>-load-<load>-runtime-progression.{pdf,png}`:
  a dedicated three-panel figure showing native solver runtime against workers,
  flows and candidate models. Individual observations distinguish timeout runs;
  a connected median curve shows progression along each dimension. Medians
  include observed timeout runtimes and pool the other scaling factors and seeds,
  as stated in the title. Missing/error runtimes are excluded and missing median
  points break the curve. The title reports observed/total completed attempts.
* `analysis/<campaign>-<objective>-load-<load>-time-to-first-incumbent.{pdf,png}`:
  observed first-incumbent times against workers, flows and candidate models,
  with timeout markers and observed medians. Missing discovery times are excluded.
* `analysis/<campaign>-<objective>-load-<load>-time-boxplots.{pdf,png}`:
  distributions for both time to first incumbent and native solver runtime,
  grouped by worker, flow and model counts. Boxes show median and quartiles;
  whiskers use the standard 1.5-IQR rule and outliers remain visible. Orange
  crosses identify observations from timeout runs. Each category reports the
  observed/total count. The other factors and seeds are pooled.
* `analysis/<campaign>-<objective>-load-<load>-wall-time-boxplots.{pdf,png}`:
  recorded `wall_time_s` distributions grouped by worker, flow and model counts,
  with the same quartiles, timeout crosses and observed/total coverage.
  Wall time covers the solver execution wrapper, including model construction,
  all incremental prefix solves, extraction and validation; it excludes earlier
  scenario profiling and subsequent journal writes. Missing historical values
  are omitted without substituting native runtime; solver-error runs are excluded.
* `analysis/<campaign>-<objective>-load-<load>-time-cdf.{pdf,png}`:
  conditional empirical cumulative distributions for each time metric, with
  separate curves for all observations, timeout runs and other runs. Every
  curve uses its own observed count as denominator. Missing observations are
  excluded and reported. Timeout runtimes are time spent before termination,
  not observations of the unknown time needed to prove optimality. A missing
  first incumbent is not assigned a discovery time at the timeout limit.
* `analysis/<campaign>-<objective>-load-<load>-time-heatmaps.{pdf,png}`:
  median times for worker/flow, worker/model and flow/model combinations, for
  both metrics. Each cell shows its median in seconds and observed/total count;
  NA marks an existing group without measurements and a dash an absent group.
  Color scales are shared across the three panels for the same metric. The
  remaining dimension and seeds are pooled.

Campaigns, objectives and loads remain separate. Missing times remain missing;
valid zero values are retained. Runtime distributions exclude solver errors and
include observed timeout runtimes. Historical runs without first-incumbent
measurements still contribute to runtime distributions and coverage counts.
`aggregated.csv` retains `first_incumbent_observed_count` and
`first_incumbent_median_s`. Analysis reads the latest attempt per run and does not
invoke the solver. Per-run incumbent curves are not generated by `analyze`;
the helper function remains available.
Previously generated files are not deleted when analysis is rerun.

## Tests and static verification

Tests cover configuration, 770/198 grids, nested reproducibility, model accuracy,
worker homogeneity, endpoint placement, links, calibration, roofline, liveness,
profiling cache/production port, accuracy eligibility, solver observations/status,
JSONL/CSV integrity, recovery, writer exclusion, aggregation and plotting.
Real profiling/solver tests require explicit opt-in:

```bash
# Future execution only; not run during this implementation.
uv run --project experiments/scalability --locked pytest experiments/scalability/tests -m 'not integration'
SCALABILITY_INTEGRATION=1 uv run --project experiments/scalability --locked pytest experiments/scalability/tests -m integration
```

Static-only verification commands and execution-path evidence are documented in
`VALIDATION.md`. No successful runtime interoperability, export, ONNX profiling,
Gurobi solve, recovery or plot generation is claimed without executing those checks.

## Bottleneck diagnostics

Bounds-only A–E ablations, shared validated references, cold/warm runs and censored
scalability analysis: [bottleneck study](bottleneck/README.md).
