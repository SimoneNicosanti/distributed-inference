# Incremental flow solver strategy

`IncrementalFlowSolverStrategy` adds flows in the original problem mapping's
insertion order. It uses the existing formulation builder, compiler, solver and
translator; both `assignment` and `cumulative` are supported. `full` remains the
default. The alternating strategy is unchanged.

## Execution

1. Solve the first flow without a seed.
2. Add one flow. Fix only prior `u`, layer assignment (`x` or native `y`), `e`,
   `r`, and the derived active/last decisions. Keep every queueing, execution,
   transmission, resource and objective auxiliary free.
3. Accept any validated feasible incumbent. Successful insertion immediately
   advances to the next flow, without an unrestricted intermediate solve.
4. Proven restricted infeasibility triggers unrestricted global repair. A
   restricted timeout without an incumbent also triggers unrestricted recovery,
   recorded separately as `restricted_timeout_without_incumbent`. Other failures
   stop explicitly with `IncrementalFlowSolveError` and stage/status diagnostics.
5. Once all flows are present, always run exactly one unrestricted final solve.
   Initialize it from the complete incremental solution. Return the better
   validated solution, respecting minimization versus accuracy maximization.
   Preserve the complete incremental solution if the final attempt times out,
   fails, has no valid incumbent, or returns a worse objective.

Every attempt owns fresh native resources. This deliberately avoids carrying
temporary bounds into repair/final models. Native search trees are not reused.
Translation and residual validation happen before model disposal. Cancellation
propagates through the existing Gurobi cancellation/draining mechanism; it is
not converted into a successful fallback.

## Original problem and warm starts

`compact_problem_prefix` shares the original validated immutable profiling maps,
worker data, links and policy. Only the flow mapping changes. The flow objects
themselves retain their identity: arrival rates, deadlines, endpoints, model
eligibility, component limits and priorities are never regenerated or rescaled.
The complete prefix objective and multiclass queueing model remain active.

`compact_solution_decisions` resolves domain flow/configuration/layer/worker/
priority identifiers before any mutation. Assignment and cumulative mappings
share this implementation. `CompactGurobiWarmStartApplier.apply_warm_start`
retains its existing exact-flow-set requirement. The explicit
`apply_partial_warm_start` method accepts a nonempty subset. It clears stale
starts, writes only prior primary decisions, and leaves new-flow decisions and
all auxiliary starts `GRB.UNDEFINED`. Rejected or unextendable MIP starts do not
add constraints and do not prevent unrestricted repair from searching normally.

`CompactGurobiIncrementalModelAdapter` uses the same mapping for temporary
variable bounds. It validates native `ConstrVio`, `BoundVio` and `IntVio` against
the model's feasibility/integrality tolerances. Numerical-failure and ambiguous
termination statuses are not accepted as feasible candidates. A prefix failure
never returns the smaller solution as if it covered the larger prefix.

Native objective values remain the comparison criterion. As with the existing
full strategy, reduced time auxiliaries can differ from recomposed reporting
times. No new optimality claim is made for a retained incremental fallback:
its `is_optimal` is false. Final native status remains independently observable.

## Selection and budgets

Optimizer-owned composition:

```python
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import GurobiStrategyConfig
from optimizer.adapters.outbound.gurobi.strategy_factory import build_compact_gurobi_solver_strategy
from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind

strategy = build_compact_gurobi_solver_strategy(
    formulation_config,
    solver_config,
    strategy_config=GurobiStrategyConfig(
        strategy=SolverStrategyKind.INCREMENTAL,
        insertion_time_limit_s=10,
        repair_time_limit_s=30,
        final_time_limit_s=120,
    ),
)
solution = await strategy.solve(problem)
```

The factory reuses `GurobiSolverConfig` and its seed, threads, gap, logging and
observer settings. A missing stage limit inherits `solver_config.time_limit_s`.
Insertion (including the first flow) and repair use `SolutionLimit=1`; proving
optimality is unnecessary there. The final solve removes that solution limit.
This addition is opt-in; existing full solves receive their original config.

Benchmark selection, from the repository root after the normal source-path setup:

```bash
export PYTHONPATH="$PWD/services/optimizer/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --python 3.14 --project experiments/scalability --locked scalability main \
  --strategy incremental --formulation cumulative \
  --output experiments/scalability/results/main-incremental
```

The existing YAML `solver` section also accepts:

```yaml
strategy: incremental
insertion_time_limit_s: 10
repair_time_limit_s: 30
final_time_limit_s: 120
```

Stage limits are **per native solve**, not a whole-strategy timeout. No additional
overall deadline was introduced. For `F` flows there are `F + 1` attempts without
repair, and at most `2F` attempts with one repair per failed insertion. Native
time limits exclude formulation/model construction. The final solve is always
attempted once a complete incremental incumbent exists, including when `F=1`.
`scalability plan --strategy incremental` reports the native solve-budget upper
bound including potential repairs and final optimization, excluding construction.

## Diagnostics and persistence

`strategy.diagnostics` exposes the latest run's immutable
`IncrementalFlowDiagnostics`; an optional strategy observer receives it also on
failure/cancellation. Every stage records flow count, fixed-flow count, reason,
native termination, validated incumbent availability, objective, construction
time (formulation preparation plus native compilation), solver runtime,
first-incumbent time, wall time and any exception details.

Summary fields include successful restricted insertions, repair count,
incremental construction wall time (first/insertion attempts, including solves),
repair wall time, final wall time, objectives before/after final optimization,
returned source/objective and strategy wall time. `native_solve_count` counts
calls into the solver port, including calls that raise.

Scalability JSONL stores the complete diagnostics; CSV adds scalar summary
columns. Run identity includes strategy and stage budgets. Aggregate analysis
separates strategies as well as formulations. Existing full plot locations
remain unchanged; incremental plots use `analysis/incremental/<formulation>/`.

For incremental runs, top-level solver runtime and construction time are sums
across attempts. Per-attempt first-incumbent times are retained in diagnostics;
no synthetic whole-strategy first-incumbent time, optimality gap or bound is
reported. Native final-attempt statistics are explicitly stored under
`last_attempt_solver_statistics`; `raw_solver_statistics` is left empty to avoid
plotting a final-attempt history as a whole-strategy history. A retained valid
fallback reports `feasible_fallback` and preserves the last solver termination
in diagnostics. Infeasibility of a restricted or incomplete prefix is not
reported as proof that the original full problem is infeasible.

## Validation and limits

New state-machine tests cover successful insertions, infeasible insertion,
timeout recovery, failed repair, mandatory final optimization, final timeout
with/without an incumbent, worse/invalid final candidates, objective senses,
empty prefixes, immutable parameter sharing and cancellation.

Native tests cover both representations, all five optimizer objectives,
multiclass queueing recomputation for prior flows, partial-start clearing,
atomic invalid-seed rejection, temporary decision bounds, an insertion that
requires moving a prior flow during repair, and a real final time limit.
Benchmark tests cover selection, independent budgets, persistence and analysis
separation. Existing tests were not edited.

Focused regression result: **196 passed** (122 optimizer, 74 benchmark), including
49 new cases. Strict mypy passed on 35 scoped source files. Ruff lint/format and
the benchmark static checker also passed; this is not a repository-wide test claim.

Validation uses small native models compatible with the available size-limited
Gurobi license. The first test fixture exceeded that license, so mapping and
solve tests use smaller instances without changing production mathematics.
Full ONNX profiling/scalability campaigns, service bootstrap and deployment were
not exercised. Fresh model construction remains a cost; persistent-model
reoptimization and overall strategy budgets are intentionally outside this change.

## Changed files

Added under `services/optimizer/src/optimizer/`:

- `application/strategies/incremental/__init__.py`
- `application/strategies/incremental/compact_problem_prefix.py`
- `application/strategies/incremental/incremental_model_adapter.py`
- `application/strategies/incremental/incremental_flow_diagnostics.py`
- `application/strategies/incremental/incremental_flow_solver_strategy.py`
- `application/strategies/solver_strategy_kind.py`
- `adapters/outbound/gurobi/entities/gurobi_strategy_config.py`
- `adapters/outbound/gurobi/strategy_factory.py`
- `adapters/outbound/gurobi/warm_start/compact_solution_decisions.py`
- `adapters/outbound/gurobi/warm_start/compact_gurobi_incremental_model_adapter.py`

Modified under that optimizer root:

- `adapters/outbound/gurobi/entities/gurobi_solver_config.py`
- `adapters/outbound/gurobi/solver/gurobi_solver.py`
- `adapters/outbound/gurobi/warm_start/compact_gurobi_warm_start_applier.py`
- `adapters/outbound/gurobi/formulation_compiler/formulations/compact/collaborators/assignment/assignment_variables.py`

Modified under `experiments/scalability/`:

- `src/scalability_benchmark/{configuration,campaigns,cli,solver,persistence,analysis}.py`
- `scripts/check_static.py`
- `README.md`

Added tests/docs:

- `test/optimizer/test_incremental_flow_strategy.py`
- `test/optimizer/test_incremental_flow_gurobi.py`
- `experiments/scalability/tests/test_incremental_strategy_selection.py`
- `docs/optimization/incremental-flow-strategy.md`

Controller, deployment, worker DTOs and dependency manifests/lockfiles were not changed.
