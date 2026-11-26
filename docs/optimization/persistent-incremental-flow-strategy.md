# Persistent incremental flow strategy

Select `incremental-persistent` to optimize successive flow prefixes using one
Gurobi environment and model. Existing `full` and `incremental` strategies remain
available; `full` remains the default.

```bash
uv run --project experiments/scalability --locked scalability-model-variant main \
  --strategy incremental-persistent --formulation cumulative \
  --output experiments/scalability/results/model-variant/incremental-persistent
```

Use `scalability-performance` for performance-based model selection. Replace
`main` with `plan` to inspect the campaign without profiling or solving. YAML:

```yaml
solver:
  strategy: incremental-persistent
  insertion_time_limit_s: 30
  # Retain the remaining ordinary solver settings.
```

The optional prefix limit overrides `solver.timeout_s`. Successful execution
still performs F unrestricted solves for F flows. Every solve inherits the
configured gap, seed, threads and logging, with no first-solution limit. The prior
solution supplies a partial MIP start, without fixing earlier decisions. There
is no separate repair or final solve. Timeout, incumbent validation, cancellation
and failure semantics match the existing unrestricted incremental strategy.

## Native model updates

The persistent compiler creates its environment/model during the first prefix,
so setup time remains part of that prefix's construction measurement. Resources
stay alive through all prefixes and close after success, failure or cancellation.
The backend-independent strategy reuses the existing incremental state machine
through the `PersistentFormulationCompiler` session port.
Fresh and persistent compilation share the same compact block assembler.
Flow-local variable maps are extended explicitly; the assignment representation
owns extension of its primary variables and derived expressions.

At each extension:

1. Add only new-flow assignment/cumulative, execution, queue-class, transmission,
   memory and response variables. Reuse worker-level variables and all previous
   native variable objects.
2. Append invariant flow-local constraints. Earlier assignment, execution,
   transmission and memory constraints remain native objects.
3. Replace the compute-queue coupling/bound block using all current flows. This
   refreshes worker SOCs, priority usage and linearizations involving queue bounds.
   Also refresh upper bounds on earlier component waits and shared queue variables.
4. Extend existing network-utilization and memory-capacity rows using `chgCoeff`.
   New memory activation inequalities reference the existing worker variables.
5. Update objective weights using the current prefix's total arrival rate.
   Tardiness/violation auxiliaries and their constraints are created only once
   per flow and then reused.

Prepared mathematical coefficients are still recomputed per prefix, including
exact queue bounds. This strategy avoids rebuilding the complete native model;
it does not eliminate all preparation or constraint-update work. Changes to the
model also do not imply preservation of the branch-and-bound tree or presolve.
No runtime speedup is assumed without measuring native construction and wall time.

Native flow indices are assigned once on insertion and retained even when the
next prepared formulation sorts flow IDs differently. Variables are merged by
domain keys, and translation/warm starts use the current prefix metadata. A
compiled prefix view is valid only inside its compilation context, before the
next extension changes the shared native model.

The extension contract is append-only: existing flow requests, profiles, workers,
links, policy, representation and utilization bins must remain unchanged. Adding
new workers/profiles or changing an existing request requires a new session.

## Experiment records and validation

Run identities, CSV/JSONL, aggregates and plot directories retain the distinct
`incremental-persistent` strategy value. Existing prefix diagnostics and solve
budget reporting apply. Construction measurements cover initial construction
or subsequent updates; native runtime sums the prefix solves. Wall-time boxplots
also include these costs. Analysis does not pool this strategy with `incremental`.
`prefix_wall_time_s` sums complete prefix attempts, including their solves;
`construction_time_s` measures preparation and native construction/updates only.

Native regression tests compare complete variable bounds/types/objective,
linear rows, quadratic rows and deadline indicators against a fresh model at
every prefix, including deliberately reordered flow IDs. They cover assignment
and cumulative formulations, all objectives, reused native objects, cancellation,
session reuse, append-only validation and benchmark integration. Tests use small
synthetic DAGs; they do not establish large-model runtime gains.
