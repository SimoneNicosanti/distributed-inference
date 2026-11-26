# Priority assignment effect: P=1 versus P=2

This independent experiment asks when two jointly optimized non-preemptive
priority classes reduce arrival-weighted mean response time relative to one
FCFS class. It does not assume a benefit. Both solves retain the same model
candidates, partitioning, placement, deadlines and resource constraints.

## Run

Python 3.14 and the existing local ONNX catalog are required. From the repository
root, use the same uv project convention as `experiments/scalability`:

```bash
uv sync --project experiments/priority_effect
export GRB_LICENSE_FILE="$PWD/gurobi.lic" # or your own working license
uv run --project experiments/priority_effect priority-effect plan
uv run --project experiments/priority_effect priority-effect pilot --output /tmp/priority-pilot
uv run --project experiments/priority_effect priority-effect run
uv run --project experiments/priority_effect priority-effect run --resume
uv run --project experiments/priority_effect priority-effect analyze
```

`--config PATH`, `--output PATH`, `--timeout SECONDS` and `--limit N` are explicit
overrides. `--retry-errors --resume` retries solver errors. `--iis` requests an
IIS `.ilp` for proven infeasibility; unsupported diagnostics retain their error
without changing the outcome. IIS work can exceed the ordinary solve budget.
`analyze --no-plots` produces tables only. Analysis never profiles or solves.

With the existing development environment, no installation is needed:

```bash
export PYTHONPATH="$PWD/experiments/priority_effect/src${PYTHONPATH:+:$PYTHONPATH}"
python -m priority_effect plan
python -m priority_effect pilot --output /tmp/priority-pilot
python -m priority_effect analyze --output /tmp/priority-pilot
```

Default catalog files live in `experiments/scalability/models`. Missing ONNX files
fail explicitly. Prepare artifacts using the existing exporter, only if needed:

```bash
uv run --project experiments/scalability --group export \
  python experiments/scalability/scripts/export_models.py \
  --models resnet50 resnet101 resnet152 efficientnet_b0 efficientnet_b2 efficientnet_b4
```

The experiment reads the base catalog and execution-efficiency schema through
`base_config`. It never runs the scalability CPU arrival calibration or its
campaign enumeration. Hardware and network overrides use the existing schemas.

## Defaults and reference deployment

`config/default.yaml` defines one CPU device, one shared edge GPU, no cloud,
device ingress and sink, and an inference-capable device. The link has
8–10 Gbit/s throughput and 0.2–0.5 ms one-way propagation. Device RAM is 8 GiB;
edge RAM/VRAM are 16/8 GiB. Hardware rates and operator efficiencies are inherited
from the synthetic profiler, with explicit device/edge hardware in the new YAML.

The main grid has flows `[2, 3, 5]`, offered loads `[0.3, 0.5, 0.7, 0.8]`, seed
`[42]`, and homogeneous/heterogeneous workloads: 24 scenarios and 48 experimental
solves, plus 24 separate reference checks. Add seeds in YAML for replication.
The pilot uses the first seed and flow count, both categories and loads
`[0.3, 0.8]`: four scenarios, eight experimental solves and four reference checks.

The objective is the existing `arrival_weighted_response_time`:

```text
J = sum_f lambda_f * T_f
reported mean response (seconds) = J / sum_f lambda_f
```

The existing native objective already divides by total arrival rate. This is a
positive scenario-wide constant, so it has the same minimizers as `J`. Preserve
`objective_value` and `objective_bound` in that native scale (seconds); do not
divide them again. `arrival_weighted_response_time_sum` records the unnormalized
`J` recomputed from extracted flows. The separately recomputed weighted mean can
differ slightly from native epigraph values within solver tolerances.

Hard deadlines remain enabled. Defaults are `max_components: 5`, compute/network
limits `0.9/0.8`, approximation relative error `0.1`, assignment formulation,
300 seconds per solve, MIP gap `0.01`, one thread and Gurobi seed `0`. All four
existing compact representations are accepted through `solver.formulation`.
Only the full solver strategy is used. Pairs differ solely in `priority_count`.

Each flow has one allowed model variant and batch size one. Homogeneous flows
use ResNet50. Heterogeneous flows use distinct catalog models: select the largest
generated service-demand contrast across ResNet and EfficientNet, then add
models maximizing distance from already selected demands. Seeded shuffling
assigns these identities to flow indices. This uses actual generated edge
profiles, not a FLOP ordering. The exact assignments and all profiled demands are
persisted. Optional exact assignments bypass selection:

```yaml
workloads:
  - name: heterogeneous
    category: heterogeneous
    models: [resnet50, resnet101, resnet152, efficientnet_b0, efficientnet_b2, efficientnet_b4]
    assignments:
      2: [resnet50, efficientnet_b0]
      3: [resnet50, resnet101, efficientnet_b0]
      5: [resnet50, resnet101, resnet152, efficientnet_b0, efficientnet_b4]
```

Arrival streams are independent Poisson streams under the existing analytical
queueing assumptions; this is not a traffic simulator. With complete-model
edge service demands `S_f` and positive relative weights `w_f`:

```text
lambda_f = L * w_f / sum_j(w_j * S_j)
sum_f(lambda_f * S_f) = L
```

Weights are sampled independently from `[0.8, 1.2]`, or supplied per flow count
through `arrival_weights: {2: [1, 2], 3: [1, 1, 1], 5: [1, 1, 1, 1, 1]}`.
Model assignment, weights, profile jitter and network draws use addressed seeded
streams. Changing P never regenerates them. The offered reference load and
optimized worker utilization are different metrics and remain separate.

The reference has three active components: device inputs, all executable model
layers on the edge GPU, device outputs. All components use class 1. Compute
waiting is `sum(lambda_f*S_f**2)/(2*(1-L))`, with deterministic service as in the
existing profiler. Transmission accounts for boundary tensors, both directed
links, production source-interface utilization bins and processor-sharing delay;
one-way propagation is added for both crossings. Deadlines are three times each
reference response, configurable through `deadline.multiplier >= 1`.

Analytic checks include compute/interface utilization, memory using production
per-flow weights plus maximum activation cut, component budget and deadlines.
They are not a certificate. By default, an additional solve pins this reference
deployment against the actual production constraints. Its native status and
certificate are persisted separately. The experimental solves receive no such
fixings. Invalid references and unsuccessful solves remain in the dataset;
arrival rates and deadlines are never repaired after observing outcomes.

## Persistence and analysis

`runs.jsonl` is an append-only, fsynced, single-writer journal reusing the existing
`ResultStore`. A committed start without a finish is retried on resume. The last
incomplete journal line can be repaired by the existing store. Source and
configuration fingerprints prevent mixing incompatible campaigns in one output
directory. Order alternates deterministically across scenario identities.

Each `scenarios/<id>/scenario.json` contains full static/execution profiles,
artifact hashes, hardware, network, assignments, weights, arrivals, final
deadlines, reference estimates and the normalized optimization input.
`reference-check.json` holds the native reference outcome; `pilot-report.json`
collects workload parameters, resource/QoS checks and paired statuses.
Software versions and source hashes are saved beside the journal.

Per-run records contain objective/bound/gap, native termination, extracted QoS,
per-flow response/deadline/tardiness, component worker/priority/layers and timing,
per-worker class loads, interface utilization, response decomposition and native
incumbent/bound histories. `both_priority_classes_used` includes zero-service
boundary components; `both_compute_priority_classes_used` counts only components
with positive service. `memory_constraint_fraction` reports native constraint
variables, which can contain epigraph slack; it is not measured memory usage.
`resource_limit_proximity` reports fractions of configured limits, not causal
proof of a bottleneck. `transmission_s` includes the model's network queueing.

Offline analysis writes `analysis/{runs,pairs,aggregates,feasibility}.csv` and
`report.json`. The legacy store's `summary.csv` is a generic solver summary;
use `analysis/runs.csv` for priority-specific fields. Nested per-flow changes
and utilization differences are JSON cells in `pairs.csv` and structured JSON
in `report.json`. It also renders six figures:

1. Paired improvement versus load, individual observations and group medians.
2. Weighted response versus load for P=1/P=2.
3. Improvement versus reference service-demand coefficient of variation.
4. Computation, compute waiting, network response, propagation and batch fill.
5. Construction/solve/first-incumbent/gap overhead.
6. All termination counts, including errors, infeasibility and missing incumbents.

Aggregates group by workload, flow count and offered load and report valid pair
and seed counts. No significance claim or independent-sample test is applied.
Unavailable objectives are excluded from numerical comparisons and retained in
feasibility counts. Negative and near-zero improvements are retained. A single
seed is a functionality pilot, not statistical evidence.

Positive improvement is `100*(Tbar_1-Tbar_2)/Tbar_1`. Runtime, construction and gap
deltas are P=2 minus P=1. Comparisons are provisional unless both gaps are at
most `tight_gap` (default 1%). Given valid minimization lower bounds `LB` and
incumbents `UB`, the true objective reduction lies in
`[LB_1-UB_2, UB_1-LB_2]`. `LB_1 > UB_2` certifies an improvement beyond incomplete
optimization. Tight-gap comparisons remain incumbent observations. A worse P=2
incumbent does not establish that a second class is intrinsically harmful.
Placement can change jointly with priorities; decomposition is diagnostic and
does not isolate a fixed-placement causal scheduling effect.

## Tests and limits

```bash
PYTHONPATH=experiments/priority_effect/src \
  pytest -q experiments/priority_effect/tests -m 'not integration'
PYTHONPATH=experiments/priority_effect/src \
  GRB_LICENSE_FILE="$PWD/gurobi.lic" PRIORITY_EFFECT_INTEGRATION=1 \
  pytest -q experiments/priority_effect/tests
ruff check experiments/priority_effect
```

Tests exercise deterministic paired inputs, calibration, assignments, deadlines,
native-status classification, bounds-aware comparison, missing values,
append/resume behavior, offline rendering and actual native execution. Native
tests also embed the same all-class-1 deployment in P=1 and P=2, checking that
the existing priority-prefix constraints permit unused class 2. No production
constraints are changed to obtain nesting.

Predictions are synthetic roofline estimates, not measurements of GPU execution
or runtime worker scheduling. Accuracy ranks are synthetic catalog metadata.
Time limits govern native optimization; profiling, compilation, translation and
optional IIS have separate overhead. A size-limited Gurobi license may reject
full DNN instances. Native errors are retained without pretending they establish
infeasibility. Default feasibility must be judged from an actual licensed pilot;
analytic checks alone do not validate the configuration empirically.

Gurobi can reject the degenerate tiny ReLU test fixture at high load for P=2
with `Q matrix is not positive semi-definite (PSD)`. This is recorded as a
solver error, not infeasibility. The experiment preserves production
`NonConvex=0`; it does not alter mathematics or enable a nonconvex solver to
bypass this error. Separate low-load native tests check joint solves and
all-class-1 nesting; high-load tests check honest error classification.

## Validation performed

On 2026-11-23, the default four-scenario pilot ran with the repository license
through `GRB_LICENSE_FILE`, the default 300-second limit and 1% gap target.
All eight experimental solves found feasible incumbents with gaps below 1%.
All four fixed reference solves returned `OPTIMAL` and feasibility certificates.
Results, reports and all six rendered figures are available locally under
`results/pilot`. A resume run left `runs.jsonl` byte-for-byte unchanged.

| Workload | Offered load | Observed P=2 response improvement | Bound assessment |
|---|---:|---:|---|
| Homogeneous | 0.3 | 0.0043% | Tight-gap incumbent comparison |
| Heterogeneous | 0.3 | 0.4013% | Tight-gap incumbent comparison |
| Homogeneous | 0.8 | 0.5121% | Certified objective improvement |
| Heterogeneous | 0.8 | 9.1906% | Certified objective improvement |

These are predicted responses from one seed and two flows, not statistical
evidence or measurements of a running distributed system. The complete
24-scenario grid was generated and passed calibration and analytic reference
checks; those additional scenarios were not solved.

Validation: 20 tests passed with native Gurobi integration enabled, Ruff check
and formatting passed, and scoped strict mypy passed for all nine source files.
The native tests include an actual IIS export, timeout without incumbent and
P=1/P=2 all-class-1 feasible-set nesting. The complete grid across multiple seeds
remains to be run.

## Extended cumulative run

`config/stress-cumulative.yaml` runs seeds `[42, 43]`, flow counts `[3, 5]`, offered
load `0.8` and both workload categories with the cumulative representation.
All other settings retain the defaults, including one allowed model per flow,
300 seconds per solve and a 1% MIP gap target. Run or resume it with:

```bash
uv run --project experiments/priority_effect priority-effect run \
  --config experiments/priority_effect/config/stress-cumulative.yaml --resume
uv run --project experiments/priority_effect priority-effect analyze \
  --config experiments/priority_effect/config/stress-cumulative.yaml
```

The completed run under `results/stress-cumulative-f3-f5-l08-s42-s43` contains
eight paired scenarios, 16 feasible experimental outcomes and eight certified
reference deployments. All P=1 solves and the four P=2 solves at three flows met
the gap target. All four P=2 solves at five flows timed out with incumbents.
Experimental native solve time summed to approximately 25.9 minutes.

| Flows | Workload | Median observed improvement | Valid pairs | P=2 final gap |
|---:|---|---:|---:|---:|
| 3 | Homogeneous | 2.42% | 2 | 0.45–0.72% |
| 3 | Heterogeneous | 8.70% | 2 | 0.83–0.93% |
| 5 | Homogeneous | 1.80% | 2 | 18.47–23.39% |
| 5 | Heterogeneous | 9.89% | 2 | 40.68–46.14% |

All eight pairs have a positive lower bound on objective improvement. The
observed percentages at five flows remain provisional estimates of the optimum
because their P=2 gaps are large. Two seeds support descriptive replication,
not a statistical significance claim. The earlier assignment run was stopped
on request; its partial results remain separate under
`results/stress-f3-f5-l08-s42-s43` with `config/stress.yaml`.

## Disaggregated cumulative run

`config/stress-disaggregated-cumulative.yaml` defines the same seed grid, flow
counts, load and solver limits, with `formulation: disaggregated-cumulative` and
`DegenMoves=-1` (automatic). The current checkout exposes only assignment and
cumulative; the disaggregated implementation is in the optimizer WIP stash
`03812c2`. This experiment uses an isolated copy of the current optimizer sources
with the stash's ten disaggregated implementation files overlaid. The current
objective and response-time builders are retained. No optimizer source in the
checkout was restored or changed.

The result directory
`results/stress-disaggregated-cumulative-auto-f3-f5-l08-s42-s43` archives the
complete optimizer copy under `optimizer-src`, the stash identity and file hashes
in `optimizer-overlay.json`, and effective Gurobi settings in
`solver-environment.json`. Its `run-with-overlay.py` launcher loads that copy and
records its actual source hashes in experiment provenance. From the repository
root, resume the selected first four scenarios (seed 42 only) with:

```bash
PYTHONPATH=experiments/priority_effect/src GRB_LICENSE_FILE=gurobi.lic \
  python experiments/priority_effect/results/stress-disaggregated-cumulative-auto-f3-f5-l08-s42-s43/run-with-overlay.py \
  run --config experiments/priority_effect/config/stress-disaggregated-cumulative.yaml --resume --limit 4
```

The recovered implementation passed 52 native formulation tests. The previous
cumulative rerun with automatic `DegenMoves` was stopped on request; its six
completed outcomes remain in
`results/stress-cumulative-auto-f3-f5-l08-s42-s43`. The earlier complete cumulative
campaign used `DegenMoves=0`, so runtime differences against it cannot isolate the
effect of formulation alone.

The user excluded seed 43 during execution. `selection.json` records the chosen
seed-42 subset; no experimental solve for seed 43 was started. The directory name
and full-grid configuration retain their original two-seed names for provenance.

All eight selected experimental solves returned feasible solutions, and all four
fixed reference deployments were certified. Native experimental solve time
totalled 11.64 minutes; six plots and the comparison tables are in `analysis/`.

| Flows | Workload | Observed improvement | P=2 final gap |
|---:|---|---:|---:|
| 3 | Homogeneous | 2.63% | 0.48% |
| 3 | Heterogeneous | 8.61% | 0.93% |
| 5 | Homogeneous | 1.91% | 23.60% |
| 5 | Heterogeneous | 8.11% | 32.41% |

Every pair certifies positive objective improvement. Both five-flow P=2 solves
timed out at 300 seconds, so disaggregation did not remove that bottleneck and
their observed improvement magnitudes remain provisional. The input problem
fingerprints match the earlier cumulative seed-42 scenarios in all four cases.

## Component priorities and objective configurations

All active components independently choose a priority on their worker through
`r[f,k,s,p]`. The supported strategies are `full_p1` and `full_p2`;
`assignment` and `cumulative` remain independent DAG
representations. The optimizer uses only component priorities.

The component configurations under `config/component-priority-*.yaml` include
response-time, normalized-tardiness and normalized-deadline-consumption runs at
3, 5, 10 and 20 flows. For 20 heterogeneous flows, explicit assignments reuse
each of ten existing models twice. Existing objective semantics and per-solve budgets are retained.

Historical experiment directories under `results/` retain their recorded
configuration, source hashes and raw outcomes. Their saved provenance describes
the code used for those runs. New runs use the current component configurations
and a fresh output directory.
