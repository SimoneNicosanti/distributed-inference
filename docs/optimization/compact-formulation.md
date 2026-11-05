# Compact MISOCP formulation

`compact` is the only supported formulation and the default. It uses the v2 epigraphs, the existing
preprocessing and operational constraints, and the existing solution interface.
No dependency was added. Existing unrelated working-tree changes were preserved.

The legacy implementation was removed on 2026-11-05 at the user's request.
The comparisons, fingerprints and test/solver measurements below are historical;
`compact-comparison.json` is retained as an archive, not an available solver path.

## Configuration

```python
from controller.adapters.outbound.optimization.gurobi_solver import GurobiSolver
from controller.domain.optimization.solving.solver_config import (
    FormulationKind,
    ObjectiveKind,
    SolverConfig,
)

settings = SolverConfig(
    objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_SLACK,
    formulation=FormulationKind.COMPACT,
)
solver = GurobiSolver(settings)
```

The existing composition root also accepts `DefaultCompositionRoot(settings)`.
Its no-argument construction now uses compact. Scalability YAML:

```yaml
solver:
  formulation: compact  # optional; compact is the default and only accepted value
  # Keep the remaining existing solver settings.
```

`SolverRunStatistics.formulation` records `compact`. An optional
observer also receives continuous/binary/nonbinary-integer counts, linear,
quadratic and general constraint counts, node count, timings, objective, final
bound and gap. Scalability records include the selected formulation and memory
assumption. Presolve reductions and root-relaxation statistics are not collected.

## Removal of the legacy implementation

The solver directly constructs `CompactFormulation`, which no longer inherits
from a legacy strategy. `FormulationKind.COMPACT` and explicit compact settings
remain supported for existing callers and experiment provenance; `legacy` is
rejected in both controller and benchmark configuration.

Removed the legacy indicator-based execution/transmission builders, memory and
PS queue modules, variable factory, per-component activation-memory field and
unused layer-activation-size helper. Shared assignment, eligibility/minimum-time,
transition-marginal, execution-queue, QoS and extraction code is retained.
All compact mathematical constraints and their active/commented status are
unchanged, including transmission/execution conservation, component activation
and the output/last-component equality. Profiles require `max_cut_size_mib`.

Legacy-only indicator/fingerprint tests were removed. Eligibility, minimum-time,
network queue and physical-response checks now target compact. Test profiles
include maximum-cut sizes; configuration checks reject legacy. No dynamic tests
or solver runs were performed for this removal. Earlier counts/timings do not
describe the current test suite or model size.

Files changed for this removal:

| Area | Files and change |
| --- | --- |
| Solver/configuration | `gurobi_formulations.py`, `gurobi_solver.py`, `solver_config.py`: compact-only strategy, construction and defaults. |
| Shared model infrastructure | `gurobi_structure_constraints.py`, `gurobi_variables.py`, `gurobi_profile_data.py`: remove only unused legacy builders, factories, activation field and helper; retained definitions have identical ASTs. |
| Compact modules | `compact_memory.py`: drop the always-empty legacy constructor field. `compact_execution.py`, `compact_execution_queue.py`, `compact_network_queue.py`: update descriptions of shared infrastructure; mathematical definitions are unchanged. |
| Deleted production modules | `gurobi_memory_constraints.py`, `gurobi_network_queue_constraints.py`. |
| Benchmark | `configuration.py`, `instances.py`, `tests/test_solver.py`, `tests/test_configuration_catalog.py`, `README.md`: compact-only validation, metadata, tests and instructions. |
| Controller tests | `test_compact_formulation.py`, `test_compact_integration.py`, `test_compact_memory_and_profiles.py`, `test_gurobi_solver.py`, `test_gurobi_time_valid_inequalities.py`; replace `test_gurobi_aggregate_indicators.py` with `test_compact_eligibility.py`, retaining eligibility/minimum-time coverage. |
| Documentation | This document; historical comparison JSON is unchanged. |

Static checks: Ruff passes on all 20 changed Python files. Targeted mypy passes
on 23 controller files and three benchmark files. AST comparison confirms the
same effective compact builder bindings and solver call order, unchanged retained
builders/factories, unchanged active/commented constraint status and byte-identical
objective, QoS, extraction, problem preparation and compute queues. All 116
optimizer imports resolve in a static source audit. Historical independent cut
and marginal reference tests remain; they provide no selectable legacy solver.

## Historical incremental implementation and validation

Each row was implemented and tested before the next subsystem. The same licensed
legacy regression suite ran after each row. Its four original failures persisted;
its passing cases and baseline fingerprints stayed stable.

| Stage | Change | Immediate validation | Optimization suite result |
| --- | --- | --- | --- |
| Baseline | Inspected preprocessing, variables, DAG/placement, eligibility, memory, queues, QoS, objective and extraction. Saved the session-start source. | Existing tests, deterministic single-worker fingerprints, objective and response. | 36 passed, 4 pre-existing failures |
| Selection | Added `FormulationKind`, default `legacy`, adapter-local strategies, observer metadata. Both strategies initially used legacy mathematics. | Configuration rejection/defaults, unchanged model sizes/fingerprints, deterministic comparisons. | 41 passed, same 4 failures |
| Integrality | Compact placements continuous in [0,1], tied to binary priority assignments; existing implicit model/activity/last/cut/usage variables reused. | Integer decisions in solved models; temporary legacy indicator promotion disappeared as those subsystems were migrated. | 42 passed, same 4 failures |
| Execution | One-sided execution bounds and finite profile-derived upper bounds. Preserved CPU eligibility, unsupported-layer exclusions, relay restrictions and execution minimum inequalities. | Fixed selected/unselected placements cannot underestimate execution; deterministic legacy comparison. | 44 passed, same 4 failures |
| Transmission/latency | Link-specific lower bounds from stored maximum cut; reused sparse transition marginals, tensor cuts and exact latency expression. Routing variables continuous and implicitly integral. | Selected/unselected source times, missing-link infeasibility, latency retained in integration. | 46 passed, same 4 failures |
| Memory | One-sided resident weights; one server activation epigraph using the v2 profiled maximum-cut proxy. Removed per-component activation variables. | Selected/unselected activation minima; deterministic comparison where memory does not bind. | 48 passed, same 4 failures |
| Priority queues | Reused existing utilization conservation, priority prefix, bins and one SOC per server/priority; added compact-only H/W bounds, bin bounds and component wait bounds. | First priority, multiple priority counts, forced high/low classes, upper-bin endpoints, actual rather than scaled waits, response-budget bounds. | 53 passed, same 4 failures |
| PS queues | Projected ordered-bin inequalities and source responses summed into each cut response; no disaggregation variables or indicators. | 20 integral/fractional cases compared with an independently constructed disaggregated LP. | 73 passed, same 4 failures |
| Aggregation/integration | Reused existing response equality, deadlines, both objectives and extraction. Added composition/scalability configuration. | Multi-model/multi-flow cases, sparse asymmetric fork/join DAG, compatibility, compute/network instability, physical timing checks. | 85 passed, same 4 failures |
| Additional verification | Added profile-chain, memory capacity/discrepancy, relay, excluded-batch, larger fingerprint and well-scaled network checks. | All implicit variables remain continuous and integral in solved compact operational cases; no compact general constraints. | 61 new optimizer tests pass in total |

Final controller + scalability suite: **248 passed, 13 failed, 5 errors, 2 skipped**.
The same 13 failures and 5 setup errors reproduce against the saved baseline
without the added tests. That comparison had 185 passes; the difference is 61 new
optimizer tests and two additional benchmark formulation cases.

The broad baseline failures comprise four optimizer failures, seven plan-publisher
failures, one deployment-builder failure and one existing `Mock(spec=gp.Model)`
test that fails before invoking the observer. Five benchmark setup errors use
obsolete profile DTO fields. They were not changed by this task. Two optional
scalability runtime tests remain skipped. Docker/Redis deployment and a full
scalability campaign were not run.

Ruff lint/format passes on all 20 changed Python files. Targeted mypy passes on
all 16 changed production Python files with `--follow-imports=silent` and
`MYPYPATH=experiments/scalability/src`. `git diff --check` passes.

Tests used the renewed repository license, without printing or changing its
contents. Reproduce the suite from repository root:

```bash
GRB_LICENSE_FILE="$PWD/gurobi.lic" \
PYTHONPATH=$(printf '%s:' services/*/src packages/*/src experiments/scalability/src) \
python -m pytest test/controller experiments/scalability/tests -q --tb=short
```

## Mathematical decisions and discrepancies

### Activation memory: explicitly authorized difference

The session-start legacy implementation uses the maximum across hosted components
of their summed layer activation sizes. v2 uses the profiled model maximum cut
for every hosted component. These are different physical assumptions. The user
explicitly selected v2 for compact and requested that legacy remain unchanged.

A regression demonstrates the difference: a chain has two 1 MB tensors, maximum
cut 1 MB, and 1.5 MB available RAM. Legacy rejects the unsplit model because its
layer activations sum to 2 MB; compact accepts its 1 MB v2 activation proxy.
Consequently, global feasible-set equivalence is intentionally not claimed for
memory-limited cases. Separate tests verify that resident weights add across
flows while activations take a maximum.

The existing profiler already produces `max_cut_size_mib`; its DTO carried that
field, but the controller mapper discarded it. The controller now retains it in
`ShapeProfile`. Compact converts MiB to bytes and then decimal MB. Missing maximum
cut data raises `GurobiProfileDataError`; no optimizer-side profiling algorithm,
tensor-sum substitution or model download was introduced.

Execution still uses the existing CPU profiles and RAM pool. No new backend
decision or GPU memory semantics were introduced.

### Existing legacy numerical behavior

1. `test_solves_feasible_single_worker_instance` reaches a stale
   `replace(..., ingress_id=...)` assertion and raises `TypeError`.
2. Two multi-variant, two-priority legacy cases raise
   `Q matrix is not positive semi-definite (PSD)` under the existing
   `NonConvex=0` settings. Test-only `Presolve=0` or `Aggregate=0` solves the same
   F=1/M=2/S=2/K=3/P=2 instance. This identifies a presolve-sensitive SOC
   recognition issue; legacy settings and constraints were not changed.
3. Legacy can suppress tiny transmission terms. Its unstable-network fixture
   returns OPTIMAL with `MaxVio=4e-6`, zero transmission times, and 4-byte tensors
   over a 1 byte/s link. Each actual transfer needs 4 seconds. Compact correctly
   rejects that instance.
4. In the F=3/M=2/S=3/K=3/P=3 comparison, legacy reports zero PS transmission
   response, while compact reports 8 microseconds per cut. This explains the
   approximately 16-microsecond response/objective difference in the recorded
   results. It was not hidden by loosening an equivalence assertion. Separate
   well-scaled 1 MB tensor tests verify both formulations against independently
   recomputed execution, first-priority waiting, PS transmission, latency and
   objective values.

The existing objective is preserved exactly: `ARRIVAL_WEIGHTED_RESPONSE_SLACK`
currently minimizes arrival-weighted response time, despite its name;
`ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_SLACK` minimizes the weighted sum of
`response / deadline - 1`. No auxiliary penalty was added.

One-sided auxiliary quantities may exceed physical minima in a feasible or
time-limited incumbent. Reducing them to physical demands preserves feasibility
and cannot worsen either existing objective. Tests reconstruct physical demands
instead of requiring equality at every feasible auxiliary assignment.

### Compact bounds and common constraints

Compact H bounds use the minimum of the per-flow maximum second-moment sum and
utilization-limit times maximum service time. Service bounds respect each model's
execution profile and remaining response budget; models excluded by batch-fill
time do not contribute. W bounds distinguish first priority, later priorities
and individual bin endpoints. The compact problem is a separate immutable copy;
shared preprocessing output is not mutated.

The shared assignment builder retains model selection, nonempty active prefixes,
DAG precedence, input/output handling, endpoint placement and the prohibition of
consecutive components on one worker. Existing sparse transition marginals forbid
missing links. Continuous routing variables remain necessary for those shared
constraints, exact latency and unchanged extraction; no additional routing
products were introduced. Shared eligibility filters, CPU profile lookup,
tensor-route/crossing expressions, objective and solution extraction are reused.

### Current follow-up status

The original compact formulation has been restored by commenting out the later
aggregate execution/transmission epigraphs, both cut-volume activation bounds,
and the priority suffix utilization bound. Their code is retained as comments.
Transmission conservation remains enabled by explicit request, including the
zero-cut constraint when no eligible link exists. The original execution minimum,
component activation constraints, queue bounds and all other original compact
constraints remain active. The output/last-component equality described below
is the only later strengthening subsequently enabled. The legacy path has since
been removed without changing these compact constraints.

This rollback received static verification only; no tests or solver were run.
The sections below preserve the history and mark each follow-up's current status.

### Follow-up: aggregate output/last-component equality (enabled)

Compact imposes `sum_m x[f,m,o_m,k] == L[f,k]` for every flow and component,
choosing the first output in each candidate model's topological order as `o_m`.
The same output is used for all components. Preprocessing already rejects models
without output layers, and the original per-output placement constraints remain.
This adds one linear equality per flow/component and no variables, including for
multi-output models. The five commented-out strengthenings remain disabled.

`compact_structure.py` first invokes the shared assignment builder and then adds
these equalities. The solver dispatches assignment constraints through the
compact formulation at the original call site, with identical arguments.
Only static verification was
performed; no tests or solver were run, and earlier model-size/timing measurements
precede this addition.

### Follow-up: transmission conservation (enabled)

Compact additionally imposes `sum_s T_tx[f,m,k,s] >= Delta[f,m,k] / beta_max[f]`
for each candidate model and cut. The maximum bandwidth is computed once per
flow using only its eligible workers and existing directed nonlocal links.
This adds one linear constraint per flow/model/cut and no variables. If no such
link exists, the constraint is `Delta[f,m,k] == 0`; there is no division by zero.
Inactive cuts already have zero crossing volume. Validity relies on the existing
prohibition of consecutive components on the same worker. Legacy is unchanged.

This follow-up received static verification only, at the user's request; neither
pytest nor a solver was executed. Earlier runtime results and the size/timing
comparison below precede this additional compact constraint.

### Follow-up: aggregate execution epigraph (disabled; code commented)

Compact additionally imposes `sum_m T_ex[f,m,k,s] >= sum_m E_ex[f,m,k,s]
- max_m E_hat[f,m,s] * (1-e[f,k,s])` for each flow, component and eligible worker.
It reuses the layer-time expressions and profile-derived full-model upper bounds
of the individual execution constraints, including CPU compatibility and relay
filtering. Only the maximum across candidate models is used as the aggregate
activation coefficient. No solver variables are added. Legacy is unchanged.
This follow-up received static verification only; the earlier runtime results
and model-size/timing measurements precede this additional constraint as well.

### Follow-up: aggregate transmission epigraph (disabled; code commented)

Compact additionally imposes `sum_m T_tx[f,m,k,s] >= (sum_m Delta[f,m,k]
- max_m Delta_hat[m] * (2-e[f,k,s]-e[f,k+1,t])) / beta[s,t]` for each cut and
eligible directed nonlocal link. Both nominal times and crossing volumes are
aggregated across candidate models; the activation coefficient is the maximum
profiled cut bound across those models. Existing individual transmission and
conservation constraints remain. No solver variables are added. Without eligible
links, the existing per-model zero-cut constraints still apply. Legacy is
unchanged. This follow-up received static verification only; earlier runtime
results and model-size/timing measurements precede this additional constraint.

### Follow-up: cut volume activation bounds (disabled; code commented)

For each flow, candidate model and existing cut, compact additionally imposes
`Delta[f,m,k] <= Delta_hat[m] * A[f,k+1]` and
`Delta[f,m,k] <= Delta_hat[m] * u[f,m]`. Both use the same profiled cut bound
and crossing-volume expression as the existing transmission constraints.
This adds two linear inequalities per flow/model/cut and no solver variables.
Legacy is unchanged. Only static verification was performed; earlier runtime
results and model-size/timing measurements precede these additional constraints.

### Follow-up: priority suffix utilization (disabled; code commented)

For each worker and priority p, compact additionally imposes
`sum_{h=p..P,f,k} rho_bar[f,k,s,h] <= rho_hat_ex * n[s,p]`, summing only flows
eligible for that worker. This reuses existing priority utilization and usage
variables and adds one linear constraint per worker/priority, without solver
variables. The existing priority prefix makes the inequality valid: when priority
p is unused, every later priority is unused; otherwise total stability bounds
the suffix. Legacy is unchanged. Only static verification was performed; earlier
runtime results and model-size/timing measurements precede this addition.

## Historical legacy preservation and shared-file audit

Four deterministic models retain their session-start Gurobi fingerprints:

| Instance | Fingerprint |
| --- | ---: |
| F1/M1/S1/K1/P1 | -2135702557 |
| F1/M1/S1/K2/P1 | -625667405 |
| F1/M1/S2/K3/P1 | -582095788 |
| F3/M2/S3/K3/P3 | -1153097706 |

All 14 pre-existing optimizer builder/data/expression/extraction modules other
than the solver orchestration are byte-identical to the session-start working
tree. Their SHA-256 hashes are recorded in `compact-comparison.json`. The baseline
includes pre-existing user edits; it is not an assumption that HEAD was clean.

Every modified production file also used by legacy:

| File | Change and reason it does not change legacy mathematics |
| --- | --- |
| `services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py` | Dispatches the same calls, arguments and order through the legacy strategy; its problem preparation is identity. Adds observational statistics only. Gurobi parameters, objective call and extraction remain unchanged. |
| `services/controller/src/controller/adapters/outbound/optimization/gurobi_formulations.py` | Assignment constraints now follow the same strategy dispatch as the other subsystems. Legacy references the original builder directly; only compact uses the wrapper that adds the output equality. |
| `services/controller/src/controller/domain/optimization/solving/solver_config.py` | Appends a default-legacy selector and defaulted statistics fields. Existing positional arguments remain compatible; `ProblemConfig` is unchanged. |
| `services/controller/src/controller/domain/profile/model_variant_static_profile.py` | Adds optional `ShapeProfile.max_cut_size_mib=None`. Legacy never reads it. Old constructors remain valid. |
| `services/controller/src/controller/adapters/outbound/profile/mappers/model_variant_static_profile_mapper.py` | Retains an already validated DTO property in the new optional field. All existing mapped values remain the same; legacy does not consume the new field. |
| `services/controller/src/controller/composition/default_composition_root.py` | Accepts an optional injected `SolverConfig`; no-argument construction uses exactly the previous defaults. |
| `experiments/scalability/src/scalability_benchmark/configuration.py` | Adds a validated selector defaulting to legacy. Existing experiment settings retain their meanings. |
| `experiments/scalability/src/scalability_benchmark/solver.py` | Passes that selector into `SolverConfig` and records it. Existing problem inputs and runtime controls are unchanged. |
| `experiments/scalability/src/scalability_benchmark/instances.py` | Makes recorded memory-assumption text accurate for compact; legacy text and instance construction are unchanged. |

New implementation files are `gurobi_formulations.py` and `compact_{variables,
bounds,structure,execution,transmission,memory,execution_queue,network_queue}.py` in the
optimization adapter. New tests are `test_compact_formulation.py`,
`test_compact_integration.py`, and `test_compact_memory_and_profiles.py` beside the
existing optimization tests. The benchmark solver test additionally exercises
both selectors; `experiments/scalability/config/default.yaml` explicitly selects
legacy. This document and the comparison JSON complete the changed-file list.

## Historical model size and measured timings

Gurobi 13.0.3; one thread; seed 0; zero target MIP gap; five repetitions.
Construction and optimization columns are medians in milliseconds. These are
tiny deterministic fixtures, not a scalability benchmark. Memory does not bind.
No speedup is claimed.

| Instance | Formulation | Continuous | Binary | Linear | Quadratic | General/indicator | Construction ms | Solve ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1/M1/S1/K1/P1 | legacy | 14 | 9 | 36 | 1 | 6 | 3.18 | 0.81 |
| F1/M1/S1/K1/P1 | compact | 14 | 8 | 40 | 1 | 0 | 4.48 | 0.81 |
| F1/M1/S2/K3/P1 | legacy | 51 | 37 | 119 | 2 | 60 | 5.03 | 1.21 |
| F1/M1/S2/K3/P1 | compact | 63 | 23 | 160 | 2 | 0 | 6.19 | 1.33 |
| F3/M2/S3/K3/P3 | legacy | 390 | 252 | 777 | 9 | 504 | 27.65 | 40.01 |
| F3/M2/S3/K3/P3 | compact | 435 | 171 | 1125 | 9 | 0 | 30.27 | 46.10 |

Largest fixture: total variables 642 to 606; binary variables 252 to 171;
general constraints 504 to zero. Linear constraints increase. All measured runs
reported a closed gap; the JSON includes objectives, fingerprints, node counts
and individual gaps. The tiny-transmission numerical discrepancy above explains
why those two largest-fixture objectives are not identical.
