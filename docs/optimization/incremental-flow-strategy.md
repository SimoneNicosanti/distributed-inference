# Incremental flow solver strategy

`IncrementalFlowSolverStrategy` solves increasingly large prefixes in the original
flow mapping's insertion order. It uses the existing formulation builder,
compiler, Gurobi solver and translator. Both `assignment` and `cumulative` are
supported. `full` remains the default strategy.

For the same prefix-solving algorithm with a native model extended in place,
select `incremental-persistent`; see the
[persistent strategy](persistent-incremental-flow-strategy.md). The `incremental`
option documented here retains fresh-model construction for comparison.

## Execution

1. Solve the complete problem for flow 1 without a MIP start.
2. Add the next flow and build the complete MISOCP for all flows in the prefix.
3. Apply the previous prefix solution only as a partial MIP start. No variable
   bounds are changed; all earlier model, layer, worker and component-priority
   decisions remain free within the original problem's domains.
4. Optimize the whole prefix to the configured gap or stage timeout. Accept a
   validated feasible incumbent even on timeout, then continue to the next flow.
5. Return the last prefix's solution, preserving its native optimality status.

The last prefix is already the complete unrestricted problem. There is no
restricted insertion, conditional repair or additional final solve. A failed
prefix without a valid incumbent raises `IncrementalFlowSolveError`; a smaller
prefix is never returned as a solution for the larger problem. Prefix objectives
are not compared to each other because they refer to different problems.

The objective, queueing and resource constraints include every flow in the
current prefix. Earlier response times and queue waits are recomputed as new
flows are added. Explicit B6 requests retain their required model; a MIP start
does not relax request eligibility or other problem constraints.

## Original problem and MIP starts

`compact_problem_prefix` shares immutable profiles, workers, links and policy.
Only the flow mapping changes. Arrivals, deadlines, endpoints and eligible
configurations are neither regenerated nor rescaled between prefix solves.

`CompactGurobiIncrementalModelAdapter.initialize_model` applies only
`CompactGurobiWarmStartApplier.apply_partial_warm_start`.
The shared `compact_solution_decisions` mapper resolves configuration, layer,
worker and priority identifiers. Only earlier primary decisions receive
`Start` values; new-flow decisions and all performance auxiliaries remain
`GRB.UNDEFINED`. All original lower and upper bounds are preserved. An infeasible
or unextendable start may be rejected or repaired by Gurobi without restricting
the search.

Every prefix owns fresh native resources. The previous solution is transferred;
the branch-and-bound tree is not. Feasibility checks and translation complete
before model disposal. Cancellation propagates through the existing native
cancellation/draining mechanism.

Native `ConstrVio`, `BoundVio` and `IntVio` are checked against feasibility and
integrality tolerances. Invalid, incomplete or numerically failed incumbents are
rejected. Successful solutions cover exactly the current prefix and have finite
objective values.

## Selection and budgets

Optimizer-owned composition:

```python
strategy = build_compact_gurobi_solver_strategy(
    formulation_config,
    solver_config,
    strategy_config=GurobiStrategyConfig(
        strategy=SolverStrategyKind.INCREMENTAL,
        insertion_time_limit_s=30,
    ),
)
solution = await strategy.solve(problem)
```

From the repository root:

```bash
uv run --project experiments/scalability --locked scalability-performance main \
  --strategy incremental --formulation cumulative \
  --output experiments/scalability/results/performance/incremental
uv run --project experiments/scalability --locked scalability-model-variant main \
  --strategy incremental --formulation cumulative \
  --output experiments/scalability/results/model-variant/incremental
```

The YAML solver section accepts:

```yaml
strategy: incremental
insertion_time_limit_s: 30
```

An omitted prefix limit inherits `solver.timeout_s`. The old repair/final
limits are removed. The factory preserves the configured threads, seed, gap and
logging while clearing `solution_limit`, so prefix solves do not stop at the
first feasible incumbent. `solver.relative_gap: 0` requests zero relative gap,
subject to the timeout and native tolerances; a timeout does not prove optimality.

For F flows there are exactly F native solves on success, including a single
solve when F=1. Their total native budget is F times the prefix time limit;
construction and translation add time. There is no whole-strategy deadline.
`plan --strategy incremental` reports that budget.

## Diagnostics and validation

Each attempt records `first_flow` or `insertion`, its flow count, native status,
validated incumbent availability, objective, construction/runtime/wall times,
time to first incumbent, errors and whether the native solver was called.
Strategy diagnostics include completed flow count, `successful_insertions`,
`native_solve_count`, and `returned_source=last_prefix` on success.
`prefix_wall_time_s` sums the complete prefix attempts, including solving;
benchmark `construction_time_s` sums only their preparation/construction times.

Benchmark runtime sums native prefix runtimes. Gap and bound are retained only
when the last native attempt covers the full requested problem. Histories and
discovery times are associated with individual attempts. Failure of an earlier
prefix does not publish its bounds as full-problem bounds.

Regression coverage includes unchanged bounds and partial starts for both
representations, previous-flow relocation without repair, all objectives,
queue recomputation, cancellation, invalid incumbents, timeout behavior,
single-flow solve count and benchmark CSV diagnostics. Native checks use small
synthetic DAGs; they do not establish scalability on real model profiles.
