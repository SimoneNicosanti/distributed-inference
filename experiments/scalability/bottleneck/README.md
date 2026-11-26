# MISOCP scalability bottlenecks

Bounds-only diagnostics of the **complete production formulation**, with component
priorities. No alternative flow-priority model or two-stage heuristic is used.
Production optimizer code and the original campaign are unchanged.

## Run

From the repository root, with the existing environment and Gurobi license:

```bash
export PYTHONPATH=experiments/scalability/src
export GRB_LICENSE_FILE=gurobi.lic
python -m scalability_benchmark.bottleneck plan --config experiments/scalability/bottleneck/config/default.yaml
python -m scalability_benchmark.bottleneck run --config experiments/scalability/bottleneck/config/smoke.yaml
python -m scalability_benchmark.bottleneck analyze --config experiments/scalability/bottleneck/config/smoke.yaml
```

`plan` performs no profiling or solves. `run --max-instances 2` limits the ordered
instance list, including reference acquisition. `--output PATH` selects a separate
results directory. `--resume` skips completed run IDs, reacquiring/revalidating
references as necessary. The measured cold B solve runs first and is reused as the
reference source. If it provides no validated witness, cold A and cold B remain
available; C–E/P and warm runs are explicitly `reference_unavailable`. Failures are journaled;
no invented placement or independently acquired per-ablation reference is used.

The default matrix has 14 instances and 140 measured solves: F=5,10,20,30,50,75,100,
seeds 42/43, L=0.8, three candidate models, K=5, two workers, P=2, cumulative DAG,
cold/warm starts and A–E. It is **not launched automatically**. Its native-solve
budget alone is 42,000 seconds, plus reference validation, separate presolve probes and
construction/profiling. Smoke runs use only F=2,3, seed 42, M=2, K=3 and 5 s per
measured solve. The optional P case is enabled in smoke.

## Configuration

`config/default.yaml` is a study overlay on `../../config/performance.yaml`.
All settings reject unknown fields. Paths resolve relative to the study YAML;
model paths resolve through the original base YAML. Important settings:

| Setting | Meaning |
|---|---|
| `flow_counts`, `seeds`, `loads` | Primary F progression, replicate seeds, nominal offered loads |
| `objectives` | Existing `ObjectiveKind` values, independently generated comparisons |
| `formulation` | Existing compact DAG representation, `cumulative` by default |
| `model_count`, `max_components`, `priority_count` | Baseline decision dimensions |
| `infrastructure_indices` | Subset of all eleven device + 0..10 edge prefixes |
| `start_modes` | `cold`, `warm`, or both; always analyzed separately |
| `include_primary_fixed` | Insert P between D and E |
| `anchor_flow_count` | F for optional secondary sweeps, default 20 |
| `candidate_model_counts`, `component_counts`, `worker_counts`, `priority_counts` | Optional OFAT lists; empty by default; workers count includes device |
| `deadline_s` | Optional predeclared deadline range, passed to the unchanged generator |
| `solver` | Native time limit, MIP gap, threads, seed, console and separate presolve probe |
| `reference.validation_seconds` | Separate original-equation validation budget; B uses the normal solver budget |

For example, add these fields to a copy of the default study to investigate
partitioning without a full factorial:

```yaml
objectives:
  - arrival_weighted_normalized_delay
  - arrival_weighted_response_time
  - arrival_weighted_normalized_response_time
candidate_model_counts: [1, 3, 5, 10]
component_counts: [3, 5, 7]
worker_counts: [1, 2, 4, 11]
priority_counts: [1, 2, 4]
anchor_flow_count: 20
```

Each secondary axis holds all other baseline dimensions constant; it is not crossed
with every primary F. Infrastructure families can be selected independently.
The main YAML and its validator currently activate only four of the eleven supported
prefixes. This study supplies its own view of the complete eleven-prefix catalog;
it neither edits the original YAML nor changes `worker_pool`/`select_workers`.

Profiles, network, model permutations, IDs, deadlines and candidate prefixes use the
existing deterministic infrastructure. The study calls the existing `flow_pool`
with `pool_size=F` for each F, so `reference_time * sum(arrival_rates) = load`.
Using a prefix of a larger calibrated pool would lower total offered load. Relative
arrival weights, IDs and deadlines remain nested; individual arrival rates rescale
with F. Realized deployment utilization can differ substantially from nominal load.
The full serialized problem, model artifact hashes, resource profiles, per-flow
candidate counts and source hash are saved with each instance.

## What is fixed

Every measured run compiles the same full problem afresh. No rows are removed or
added, and no objective coefficients or variable types are changed. Only selected
LB/UB and optional MIP-start attributes change. Original and post-fixing dimensions
are separate because fixed binary variables still count in `NumBinVars`.

| Case | Compiler containers fixed via LB=UB |
|---|---|
| A | None |
| B | `assignment.configuration_selected`, set to variants predeclared per flow |
| C | B + `assignment.layer_decision_items()` + `component_active` + `component_last` |
| D | C + `assignment.component_worker` |
| P, optional | D + `compute_queue.component_priority` |
| E | All primary decisions above + every remaining native binary/integer variable |

`layer_decision_items()` handles direct x and cumulative y representations without
fixing affine layer-assignment expressions. Native `u`, activation, `last` and
placement variables may have type **continuous but be logically binary** through
existing equations. Their role, not `VType` alone, determines primary fixing. Their
reference values are rounded only within `IntFeasTol` and then revalidated.

E includes queue-discretization and other auxiliary integers. All physical continuous
quantities—execution, utilization, queueing delays, response times, etc.—retain their
original bounds and are recomputed. `free_integer_variables` counts native unfixed
B/I domains; `implied_binary_continuous` records the logical-decision distinction.
Without P, **D→E combines priorities and auxiliary discretization**. E conditions on
reference discretization; it does not measure an unrestricted continuous queueing
model or isolate queueing from the remaining network/resource constraints.

## Common feasible reference

For each exact source/problem/formulation identity:

1. **Before any solve**, predeclare one eligible DNN variant per flow using the
   existing addressed stream `stream(instance_seed, "model-selection", flow_id)`.
   Choices do not depend on A, deadlines, arrival rates, objective or infrastructure
   when the available model prefix is unchanged. The synthetic benchmark has one
   input configuration per variant; ambiguous multiple-configuration variants are
   rejected rather than silently fixing additional batch decisions. Save each
   flow's chosen variant/input configuration in the instance and run metadata.
2. Run **B cold first**, fixing only these model choices on the complete model.
   Use the same time limit, gap, threads and solver seed as A. With the default
   preset this means **300 seconds**, not a separate 60-second acquisition cap.
   There is no first-incumbent `SolutionLimit=1`. A feasible timeout incumbent is
   sufficient to attempt validation; B need not reach optimality.
3. Normalize the discrete native values within `IntFeasTol`. Validate by compiling
   the original complete model, fixing the discrete decisions as in E, and
   recomputing physical continuous values. Require `BoundVio`/`ConstrVio <=
   FeasibilityTol` and `IntVio <= IntFeasTol`. The reference must agree exactly with
   the predeclared model choices.
4. Save the validated B-derived witness and content ID. C, D, P and E use it for
   their additional bounds. All warm runs use this **same full native assignment**
   as a MIP start. **A warm still has all model/partition/placement/priority bounds
   free**; B warm retains its predeclared models. A's results never determine the
   model choices or the reference.

B cold is executed **once**: its existing native metrics are also the measured cold
B result. It is not rerun after obtaining the reference. Its complete result is
cached, even on timeout without an incumbent; resume revalidates a saved candidate
without rerunning B. Changed source, instance, solver/validation settings, model
assignment or reference policy get a separate identity. Legacy A-derived reference
caches are rejected. Reports/analysis label `predefined_models_from_b_v1` separately
from the historical full-A reference protocol.

If only warm mode is requested, cold B remains necessary as a separate preparation
solve, outside the requested warm results; `plan` reports its extra budget. Otherwise
B's runtime is already counted among measured solves. Validation, model construction,
presolve probes and reference-preparation costs are separately recorded. A warm's
native runtime excludes B preparation; this diagnostic is not an end-to-end speedup
claim. If predefined B is infeasible, cannot find an incumbent, or fails validation,
A cold still runs, B cold retains its actual outcome, and dependent/warm cases are
unavailable. No alternative model is selected after observing solver outcomes.

The obsolete `reference.acquisition_seconds` field has been removed; remove it from
custom YAMLs. B always uses `solver.time_limit_seconds`. Current preset outputs end
in `-b-reference`, preserving previous campaigns and preventing accidental mixing.

## Measurements and output

Under the selected output directory:

- `manifests/*.json`: full effective study/base settings, versions, dirty-source hashes.
- `instances/*.json`: exact original problem and scenario metadata.
- `references/*.json`, `*-report.json`: common witness, acquisition/validation evidence.
- `logs/*.log`: one native Gurobi log per measured run.
- `runs.jsonl`: fsynced, locked, resumable raw journal, including failures/unavailable cases.
- `summary.csv`: standard scalability summary; the complete diagnostic CSV is
  **`analysis/runs.csv`**, including nested dimensions, native parameters and histories.
- `analysis/aggregate.csv`, `transitions.csv`, `thresholds.json`, `summary.md`.
- `analysis/plots/*.{png,pdf}`: runtime, first incumbent, success, gap, presolved binaries,
  first root relaxation, conditional transitions and representative runtime breakdown.

Dimensions include original/post-fixing/probe-presolved counts, free B/I counts,
linear/Q/general rows, nonzeros and SOC rows identified by the production queue-SOC
names. A separate `native.copy().presolve()` probe supplies presolved dimensions
without modifying the measured model. It has an explicit budget and is not part of
measured runtime. Its result can differ from optimize-time presolve because Gurobi's
processing context differs. Log-derived presolved counts are also retained.

Production callbacks collect incumbent and bound traces, first feasible and first
zero-objective times. A continuous/fixed solve may emit no MIPSOL callback; its time
to first incumbent then remains missing even if a final solution exists. Native
status/runtime/nodes/objective/bound/gap are retained. Logs provide presolve time and
the **first** root relaxation time/objective when printed, not a claimed final root
bound. Missing/not-reached phases are null. Logs round times; no exact decomposition
is inferred. Unavailable breakdowns show total duration with a hatch.

All measured ablations use identical settings (defaults: 300 s, gap 0.01, one thread,
seed 0). Effective native tolerances and inherited `gurobi.env` tuning are recorded;
there is no per-ablation tuning. `solved_to_tolerance` is distinct from a closed-gap
global-optimality observation. Native feasibility-quality measurements remain visible
even when the final incumbent marginally exceeds a nominal tolerance.

Normalized delay is positive normalized tardiness, with zero lower bound; a fast zero
can establish optimality without exploring the general search difficulty. Normalized
response is the mean arrival-weighted fraction T/D and preserves hard deadlines.
The study retains the production objective/deadline semantics. Predeclared workload
and deadline settings may naturally produce positive tardiness; outcomes never cause
workload changes.

## Interpretation

Summaries separate cold/warm, objective, formulation, infrastructure, OFAT axis and
all other fixed dimensions. Runtime medians/IQR use **solved-to-tolerance runs only**;
observed budget durations, success/timeout/feasibility rates and gap distributions are
reported alongside them. Raw records include `mip_start_requested`, log-observed `mip_start_loaded` and the
loaded objective, so a supplied start is distinguished from one Gurobi actually
loaded. A "did not produce a new incumbent" message is ambiguous and is not treated
as proof of infeasibility. Callback-time medians use observed events only and include
their sample counts. Missing measurements are never imputed as zero.

Transitions are paired by original instance and reference. Runtime ratios/differences
are computed only if both solves reached configured tolerance. A timeout is censored,
not an exact time to optimality. No timeout/completed-solve speedup is computed.

The earliest tested difficulty flag is the first primary-F point with timeout fraction
at least 0.5, or solved-subset median runtime at least 5x the preceding tested F.
These are declared descriptive rules, not a statistical changepoint test; single-seed
and selectively solved subsets require caution. The largest paired median reduction
is a conditional observation, not an additive attribution or a model-equivalence claim.

Fixing to one feasible, possibly suboptimal witness changes both feasible region and
best objective. Reference bias, interactions, auxiliary discretization, presolve and
small samples limit causal interpretation. B→C is partitioning **conditional on the
predeclared per-flow models**. E includes all remaining continuous constraints, so its
runtime alone cannot identify a particular queueing equation as the cause.

## Validation

Focused native/experiment regression command (repository root):

```bash
GRB_LICENSE_FILE=gurobi.lic PYTHONPATH=experiments/scalability/src:. python -m pytest -q \
  experiments/scalability/tests/test_bottleneck.py \
  experiments/scalability/tests/test_bottleneck_b_reference.py \
  test/optimizer/test_component_priority_regression.py \
  test/optimizer/test_cumulative_formulation.py \
  test/optimizer/test_normalized_response_objective.py \
  experiments/scalability/tests/test_persistence_analysis.py \
  experiments/scalability/tests/test_incumbent.py \
  experiments/scalability/tests/test_component_priority_contract.py
```

**106 focused native/experiment regression tests pass.** They cover: exact variable-role fixing for
both DAG representations, unchanged physical continuous bounds and restored model
signatures, B-derived reference validation/cache rejection, independent/nested
predefined variants, free model selection in A warm, single execution of B cold,
all three main objectives,
naturally positive tardiness in a predeclared native test workload, native infeasible
and time-limited statuses, unchanged scenario inputs, unavailable-reference
baseline preservation, and presolve cancellation joining its native thread before disposal. Reporting tests verify censoring, missing measurements and
all eight plot families. Existing component/full/incremental contracts and native
matrix regressions are included; this is not a full-repository test run.

Ruff and strict project mypy pass for the new package:

```bash
ruff check experiments/scalability/src/scalability_benchmark/bottleneck experiments/scalability/tests/test_bottleneck*.py
MYPYPATH=experiments/scalability/src mypy --follow-imports=silent --ignore-missing-imports experiments/scalability/src/scalability_benchmark/bottleneck
```

Runner resume tests verify that cached B measurements are reused and completed
measured solves are not rerun. Generated profile caches
are source/version keyed and ignored by Git. See [PILOT.md](PILOT.md) for historical full-A-reference
pilot evidence; those results do not measure the new B-reference protocol.

## Files

Created:

- `src/scalability_benchmark/bottleneck/__init__.py`, `__main__.py`: package and CLI.
- `configuration.py`, `instances.py`: validated overlays, OFAT plans and unchanged generator views.
- `model_selection.py`, `decisions.py`, `reference.py`: predeclared variants, variable-role fixing and B-derived validated references.
- `solver.py`, `metrics.py`: production invocation, native/probe/log measurements and cancellation cleanup.
- `runner.py`: locked journaling, provenance, reference reuse and baseline preservation.
- `analysis.py`, `plots.py`: censored summaries, conditional pairs and eight figure families.
- `tests/test_bottleneck.py`, `tests/test_bottleneck_b_reference.py`: diagnostic/native/reporting tests.
- `bottleneck/config/{default,smoke,pilot,pilot-reference60}.yaml`: study presets.
- `bottleneck/{README,PILOT}.md`, `bottleneck/.gitignore`: instructions, measured pilot and artifact exclusions.

Paths beginning `src/` and `tests/` are relative to `experiments/scalability/`.
Modified only the parent `README.md` (study link) and `.gitignore` (generated profile
cache exclusion), in addition to the new files. Preexisting worktree changes are
preserved. No production `services/` or `packages/` files were changed by this study.
