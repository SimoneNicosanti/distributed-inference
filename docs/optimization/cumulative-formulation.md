# Assignment and cumulative compact MISOCP

The optimizer owns both representations. `assignment` remains the default;
`cumulative` changes the layer representation, precedence rows and tensor-prefix
expressions. Controllers, optimization problems, DTOs and public solutions are
unchanged. Execution, memory, placement, routing, priority queues, PS queues,
objectives, deadlines and strengthening constraints use the same collaborators.

## Selection and implementation

```python
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import CompactFormulationKind
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import MisocpCompactFormulationConfig

config = MisocpCompactFormulationConfig(
    relative_error=0.1,
    representation=CompactFormulationKind.CUMULATIVE,
)
```

Use this config with the existing `MisocpCompactFormulationBuilder` and
`GurobiMisocpCompactFormulationModelBuilder`. The prepared formulation records the
representation. The native model builder selects `CompactAssignmentBuilder` or
`CumulativeAssignmentBuilder` before creating any assignment variables.
There is no model conversion or postprocessing of an assignment model.

`AssignmentVariables.layer_component` is a solver-local view of affine terms:
native `x` variables for assignment, derived expressions for cumulative.
`CumulativeAssignmentVariables` owns the actual `y` variables, prefix access and
MIP-start mappings. Shared mathematical collaborators consume the affine view.

## Integer equivalence

For each flow/candidate/layer, cumulative creates exactly `K-1` binary variables,
with fixed expressions `y[0]=0`, `y[K]=u`. It explicitly makes `u` binary and adds
`y[k] <= y[k+1]`. The assignment is the expression `y[k]-y[k-1]`.

If `u=0`, monotonicity and nonnegative binary domains force every `y` to zero.
If `u=1`, the monotone binary sequence switches from zero to one exactly once.
Its differences therefore assign the layer to exactly one component and
telescope to `u`. For `K=1` there are no `y` variables and the assignment is `u`.

Conversely, each original integer assignment maps uniquely to its cumulative
prefixes. Every original precedence row is the same inequality on those
prefixes; cumulative emits only `y[target,k] <= y[source,k]` for `k<K`.
The final `k=K` row is an identity. No total ordering, candidate cut restriction
or grouping of operators is added. Incomparable DAG branches remain free to
exchange component order.

Tensor crossing uses direct prefix differences. Single-maximal-consumer tensors
introduce no crossing variable. Multiple maximal consumers keep the original OR
variable domain and all its lower/upper constraints, with direct `y` terms.
All other assignment consumers use the difference expressions. Thus weights,
execution costs, component nonemptiness, endpoints and conservation inequalities
are preserved under the bijection. This establishes integer equivalence. The
unpresolved continuous relaxations correspond under the same substitution;
presolve, generated cuts, branching and numerical behavior can differ.

The default assignment path retains its original variable domains, rows and
coefficients. Four pre-change model fingerprints (one/two candidates, `K=1/3`)
matched after implementation, including variable order, bounds, objective,
linear rows and quadratic constraints.

## Solutions and starts

The translator evaluates affine assignments, then returns the same
`OptimizationSolution` objects. Physical execution remains recomputed from
selected layer profiles, as required by the reduced-time policy. Shared queue
estimates and native objective/status retain their existing meanings.

Warm starts clear stale starts before applying validated decisions. Assignment
sets `x[layer,k]`; cumulative sets every `y[layer,cut]` with `cut >= k`. Unselected
candidate variables start at zero and auxiliary starts remain undefined. This
supports transfer in both directions and a smaller-component seed into a larger
cumulative model. The incomplete alternating-strategy orchestration is not
changed by this implementation.

## Scalability

From the repository root, after the normal experiment setup:

```bash
export PYTHONPATH="$PWD/services/optimizer/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --python 3.14 --project experiments/scalability --locked scalability main --formulation assignment --output experiments/scalability/results/main-assignment
uv run --python 3.14 --project experiments/scalability --locked scalability main --formulation cumulative --output experiments/scalability/results/main-cumulative
```

Both commands also accept the existing `--objectives`, `--resume`, `--limit` and
`--analyze` options. YAML `solver.formulation` accepts `assignment` or `cumulative`;
the old `compact` value is an alias for `assignment`. The grid, eleven
infrastructures, seed policy, workloads, deadlines, component limits, profiles
and Gurobi parameters are unchanged.

For a paired comparison, generate each scenario once, alternate representation
order and retain all native outcomes:

```bash
uv run --python 3.14 --project experiments/scalability --locked python experiments/scalability/scripts/compare_formulations.py --limit 1 --repetitions 3 --output experiments/scalability/results/paired-formulations
uv run --python 3.14 --project experiments/scalability --locked scalability analyze --output experiments/scalability/results/paired-formulations
```

`--limit` counts base main-grid instances. A single instance with three
repetitions produces six solves. This command needs the normal model artifacts,
profiling dependencies and an adequate Gurobi license. It was not used to run a
real-model campaign during this implementation.

Formulation enters `RunSpec`, run IDs, JSONL, CSV and aggregation keys. Graphical
artifacts are separated into `analysis/assignment/` and `analysis/cumulative/`.
Historical records without a formulation remain `unspecified`; the historical
`compact` value maps to `assignment`. Records are never silently pooled across
representations.

## Metrics and limits

Existing timing, incumbent, objective, bound, gap, node and status observations
are retained. Additional scalar CSV/JSONL fields include original variable,
binary and constraint counts, linear nonzeros and quadratic-constraint nonzeros.
These are separate Gurobi counts, not a synthesized combined matrix size.
Bound traces and full incumbent traces are retained in raw JSONL statistics.
Presolve column/row removals are observed through callbacks. Bound observations
are limited to one per integer second plus a final snapshot. Zero-tardiness
discovery is reported only for deadline-violation/normalized-delay objectives,
from the first MIPSOL objective within `1e-9` of zero.

These observations do not expose the exact presolved matrix size/nonzero count
or isolate root-relaxation runtime. No extra `model.presolve()` call is made: it
would add work and confound the comparison. Use native logging (`output_flag:
true`) for root and presolved matrix diagnostics. Callback definitions and model
attribute semantics follow the [Gurobi callback reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/callbacks.html)
and [model attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html).

## Validation and measured comparison

New optimizer tests cover `K=1/2/3/7`, pure binary `y` with no `x` variables,
binary selection, direct two-coefficient precedence/tensor-prefix rows, inactive
models, multiple outputs, branching and skip edges. Exhaustive testing checks
276 candidate layer partitions per representation (41 feasible with unrestricted
endpoints), including all tensor-cut indicators and byte volumes. Native full
tests cover all five objectives, single/multiple flows and candidates, CPU/GPU
workers, one/two priorities, batch sizes one/four, fixed endpoints, deadlines,
constraint residuals and cross-representation MIP starts. Matching a timeout
incumbent is never used as proof: objective comparisons require `OPTIMAL`.

Final focused validation: **79 optimizer tests and 68 experiment tests passed**
(147 total). Mypy passed on 65 production/script files; Ruff lint and formatting
passed on the scoped source/test/script set. The static/grid checker and CLI
`plan --formulation cumulative --objectives all` passed. Existing test files
were not modified. Historical multi-shape, exact-auxiliary and reverted
independent-budget tests remain outside this passing-suite claim, as requested.

The standalone [measurement artifact](cumulative-formulation-measurements.json)
contains five repetitions per representation on one small deterministic fixture:
one flow, one eight-layer branching/skip DAG, two outputs, two workers,
`K=3`, `P=1`, one solver thread, seed zero, zero relative gap and ten-second limit.
All ten solves returned `OPTIMAL`. The artifact records the Gurobi version and
a whitelist of effective solver parameters, including inherited environment
settings; it contains no license parameters.

| Quantity | assignment | cumulative |
| --- | ---: | ---: |
| Original variables | 105 | 97 |
| Binary variables | 36 | 29 |
| Linear constraints | 236 | 234 |
| Linear nonzeros | 865 | 801 |
| Median construction time | 3.775 ms | 3.517 ms |
| Median native runtime | 7.448 ms | 3.482 ms |
| Objective | 0.007096885 | 0.007097528 |

The objective difference is approximately `6.43e-7`, within the native numerical
tolerances and the tests' comparison tolerance. This small timing sample does
not establish a scalability improvement. Both models use identical settings;
the new representation reduces prefix sparsity overhead but can densify
execution and memory expressions.

Reproduce this fixture without loading ONNX models:

```bash
PYTHONPATH="$PWD:$PWD/services/optimizer/src${PYTHONPATH:+:$PYTHONPATH}" uv run --python 3.14 --project experiments/scalability --locked python test/optimizer/measure_cumulative_formulations.py --repetitions 5 --output /tmp/cumulative-formulation-comparison.json
```

Production changes are in optimizer configuration/preparation, the assignment
variable view and builders, precedence/tensor expressions, model-builder
dispatch, translation/MIP-start mappings and solver observations. Experiment
changes are in selector/CLI/spec metadata, reporting/analysis and the comparison
script. No controller, deployment, runtime-inference or dependency changes are
required.

## File manifest

Modified:

- `experiments/scalability/README.md`
- `experiments/scalability/config/default.yaml`
- `experiments/scalability/scripts/check_static.py`
- `experiments/scalability/src/scalability_benchmark/analysis.py`
- `experiments/scalability/src/scalability_benchmark/campaigns.py`
- `experiments/scalability/src/scalability_benchmark/cli.py`
- `experiments/scalability/src/scalability_benchmark/configuration.py`
- `experiments/scalability/src/scalability_benchmark/persistence.py`
- `experiments/scalability/src/scalability_benchmark/solver.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/entities/gurobi_run_statistics.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/formulation_compiler/formulations/compact/collaborators/assignment/assignment_variables.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/formulation_compiler/formulations/compact/collaborators/assignment/constraints.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/formulation_compiler/formulations/compact/collaborators/network/expressions.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/formulation_compiler/formulations/compact/gurobi_misocp_compact_formulation_compiler.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/result_translation/formulations/compact/gurobi_misocp_compact_solved_compiled_formulation_translator.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/solver/gurobi_solver.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/warm_start/compact_gurobi_warm_start_applier.py`
- `services/optimizer/src/optimizer/application/formulation/compact/misocp/misocp_compact_formulation_builder.py`
- `services/optimizer/src/optimizer/domain/formulation/compact/misocp/misocp_compact_formulation.py`
- `services/optimizer/src/optimizer/domain/formulation/compact/misocp/misocp_compact_formulation_config.py`

Added:

- `docs/optimization/cumulative-formulation-measurements.json`
- `docs/optimization/cumulative-formulation.md`
- `experiments/scalability/scripts/compare_formulations.py`
- `experiments/scalability/tests/test_cumulative_selection.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/entities/gurobi_bound_sample.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/formulation_compiler/formulations/compact/collaborators/assignment/cumulative_assignment_builder.py`
- `services/optimizer/src/optimizer/adapters/outbound/gurobi/trackers/gurobi_progress_tracker.py`
- `services/optimizer/src/optimizer/domain/formulation/compact/misocp/compact_formulation_kind.py`
- `test/optimizer/measure_cumulative_formulations.py`
- `test/optimizer/test_cumulative_compatibility.py`
- `test/optimizer/test_cumulative_formulation.py`
