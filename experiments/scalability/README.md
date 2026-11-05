# Scalability benchmark

Synthetic, nested optimization instances. Reuses the controller's `GurobiSolver`
and the model manager's complete `DefaultModelProfiler`. No microservices,
Redis, HTTP endpoints, worker processes or GPU hardware are needed to construct
the simulated infrastructure. Real Gurobi execution needs an adequate license.

The solver uses only the compact formulation. `solver.formulation` defaults to
`compact`; the explicit value `compact` remains accepted and `legacy` is rejected.
Profiles must contain `max_cut_size_mib`. Historical result files remain readable.

## Setup

Run commands from the repository root. Python 3.14 or later is required.

```bash
uv sync --project experiments/scalability --locked
uv run --project experiments/scalability --locked scalability plan
```

The project has its own manifest and lockfile. Production manifests and lockfiles
are unchanged. Plotting uses matplotlib; exporting additionally needs the export
dependency group. These commands are provided for subsequent execution: no
dependencies, exports, profiling, plots, tests or solves were run during the
static-only implementation verification.

The export group resolves `torch` and `torchvision` exclusively from the explicit
PyTorch CPU index. It does not install CUDA, NVIDIA runtime packages or Triton.

## Explicit model preparation

Place all six ONNX files in `experiments/scalability/models/`, or prepare them:

```bash
uv sync --project experiments/scalability --locked --group export
uv run --project experiments/scalability --locked --group export python experiments/scalability/scripts/export_models.py
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

The script uses eager export (`dynamo=False`), FP32, static batch 1, opset 17,
input `images`, output `logits`, and deterministic unique ONNX node names.
It then invokes the existing `OnnxModelOptimizer` at BASIC optimization level.
This keeps standard ONNX operators; extended provider-specific fusions may not be
supported by the existing static profiler. Raw exports remain in `models/source/`.
Each final artifact has an `.export.json` manifest with export settings, package
versions and artifact hashes. Shape, export seed, opset and weight mode are YAML
settings, with explicit command-line overrides. Keep the local weight files and
export manifests when archiving an experiment.

Default image sizes: ResNet-50/101/152 and EfficientNet-B0 224x224,
EfficientNet-B2 260x260, EfficientNet-B4 380x380, three channels.
Dynamic symbols in manually supplied models receive exactly one configured value.
Contradictory static dimensions and anonymous dimensions fail validation.
The existing profiler currently requires input `images`; it is not modified.

## Commands

```bash
# Six representative runs, native solver timeout 60 seconds, then offline analysis.
uv run --project experiments/scalability --locked scalability pilot --output experiments/scalability/results/pilot --analyze

# Main: seed 42, load 0.6, 5 flow counts x 6 model counts x 11 infrastructures.
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

Main: **330** configurations, 300-second timeout and relative gap **0.01**.
Sensitivity: **198** additional configurations, loads 0.3/0.8, flow counts 1/3/5,
model counts 1/3/6, all 11 infrastructures. Solver time budgets: 27.5 and 16.5
hours, 44 hours combined. Model construction, profiling, persistence and solver
shutdown add time; these figures are not strict wall-clock bounds.

## Synthetic assumptions

All assumptions are in `config/default.yaml`; changes are included in result IDs.
There is no automatic calibration adjustment, deadline relaxation or feasibility
repair. Report pilot findings before changing assumptions.

* Independent addressed random streams: model order, synthetic accuracy, layer/tier
  execution jitter, worker-pair networking, flow arrival weights, and deadlines.
  They use SHA-256-derived seeds, not Python's randomized `hash()`.
* Accuracy increases strictly with the explicit global YAML ranking. This ranking
  is synthetic, not measured ImageNet performance. Candidate sets are prefixes of
  a separate seeded permutation. `PerformanceMetricType.ACCURACY` uses `>=` with
  matching dataset; every flow's minimum is zero.
* One full scenario is generated per seed: six models, five flows, eleven workers
  and 110 directed links. Instances select prefixes/subsets without resampling.
  Worker IDs remain stable, and identical tiers share execution profiles.
* Infrastructure: device alone, one through five edges, then one through five
  clouds while retaining all five edges. Never an edge/cloud Cartesian product.
* Device: four CPU cores, 8 GiB RAM. Edge: 8 GiB VRAM, 16 GiB physical RAM.
  Cloud: 24 GiB VRAM, 64 GiB physical RAM. Edge/cloud host core counts are explicit
  synthetic assumptions (8/16), not parallel service multipliers.
* **Authorized solver projection:** GPU roofline times occupy `Backend.CPU`,
  because the current formulation selects this slot only. GPU VRAM occupies the
  solver's `ram_profile` capacity (8/24 GiB). Device uses its actual 8 GiB RAM.
  Physical RAM, simulated backend and projected capacity remain separate metadata.
  This is a benchmark adapter convention, not production GPU-backend support.
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
* Five Poisson flows have independently sampled weights in [0.8, 1.2]. Reference
  service time is the complete ResNet-50 CPU model on the device, always available
  for calibration even if absent from a candidate prefix. For load `L`:

  `lambda_i = L * weight_i / (reference_service_time * sum(all_five_weights))`.

  Thus all five reference flows together offer utilization `L`. Smaller prefixes
  do not rescale rates. Sensitivity scales arrivals only: IDs, deadlines, models,
  execution, workers and networking remain identical for a seed.
* Deadlines are uniform in [0.2, 5] seconds. Ingress and sink are fixed to the
  device, which is also an inference runner. The required finite energy limit is
  explicit YAML metadata; the current formulation does not enforce energy.
* Seven components use the existing formulation's component definition, including
  its input/output boundary layers. Priorities, queue discretization, objective,
  utilization constraints and transmission accounting are reused unchanged.

## Solver evidence and reproducibility

The optional `SolverConfig.observer` receives immutable `SolverRunStatistics`
before the native model closes, including error exits from `optimize()`.
Existing callers without an observer keep the same solution interface.
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
time are distinct. Construction includes preparation and environment startup.
Time to first incumbent is not collected.

Source hashes include dirty production files, the experiment and its lockfile.
Artifact hashes include ONNX external-data sidecars. Cached static profiles are
keyed by artifacts, shape, accuracy, source and dependency versions. Run IDs also
include effective scenario configuration, campaign, timeout and software/host
identity. Changing any of these starts a new set of runs rather than reusing
incompatible results. Fixed instances are reproducible; wall-clock-limited solver
results and runtimes need not be identical across repetitions.

References: [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html),
[Gurobi parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html).

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

## Tests and static verification

Tests cover configuration, 330/198 grids, nested reproducibility, model accuracy,
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
