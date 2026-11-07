# Static-only validation

Implementation verification deliberately did **not** execute pytest (including
collection), benchmark modules, torchvision export, ONNX profiling, Gurobi solves,
pilot/campaign execution, plots, Docker or microservices. No environment sync or
package installation was performed. The experiment lockfile was resolved and
checked; this validates dependency resolution, not installation or runtime.

## Aggregate time distribution verification (2026-11-06)

Offline analysis now writes three additional PDF/PNG figures for each
campaign/objective/load: grouped boxplots, conditional empirical CDFs and pairwise
median heatmaps. Each covers both time to first incumbent and native runtime.
Observed/total counts make missing measurements explicit. Valid zero times are
retained; runtime observations from solver errors are excluded, while timeout
runtimes remain measured termination times. CDFs use the observed count within
each labeled subset and do not infer time to optimality or impute a missing first
incumbent at the timeout limit. Heatmaps pool the remaining scaling dimension,
share a color scale per metric and distinguish absent cells from cells with no
measurements. Time-to-first-incumbent progression output is disabled; the
already-commented per-run incumbent plotting loop remains disabled.

Static verification covers the new offline entry point and compilation of 41
Python files, plus Ruff lint/format, focused strict mypy on five changed Python
files and `git diff --check`. Test definitions cover zero/tied CDF samples,
missing/invalid measurements, timeout/error treatment, pairwise coverage,
objective-separated file names and the absence of first-incumbent progression
and per-run curve output. No tests, benchmark functions, plotting or solver runs
were executed. Appearance and runtime behavior remain unverified.

## Runtime progression verification (2026-11-06)

The existing runtime panels displayed individual points only. Offline analysis
now also writes a dedicated `*-runtime-progression.pdf` and PNG per
campaign/objective/load, with worker/flow/model panels and connected observed
median curves. Timeouts retain their measured runtimes and use distinct markers;
the median includes them. Other factors and seeds remain pooled, explicitly
stated in the title, so the curve describes that mixture rather than a comparison
at fixed values of every other factor. Missing/error runtimes are excluded;
missing medians become gaps and coverage is reported in the title.

Ruff lint/format, focused strict mypy on the three changed Python files, the
39-file syntax/grid/solver-flow checker and `git diff --check` passed. Static
checks cover the new entry point in `analyze`. Existing plotting test definitions
now expect the extra PDF/PNG files and check objective-separated file names;
the aggregation contract retains timeout runtimes in the median. No tests,
benchmark execution or plot rendering were run. Appearance and runtime behavior
remain unverified.

## Incumbent plotting verification (2026-11-05)

`analyze` now adds a time-to-first-incumbent figure per campaign/objective/load,
with worker/flow/model panels, individual observations, timeout markers, observed
medians and coverage counts. Aggregated CSV rows expose the corresponding count
and median. Each run with a usable incumbent history also produces its own
PDF/PNG step curve under `analysis/incumbents/<run_id>.*`, including the first
discovery marker and final fractional-second sample. Missing values become plot
gaps; valid zero objectives and discovery times remain observations. Historical
runs without the measurements are skipped for the relevant new figures.

Ruff lint/format, focused strict mypy on the three changed Python files, the
39-file syntax/grid/solver-flow checker and `git diff --check` passed. AST checks
verify both plotting entry points are called from offline analysis and the
incumbent curve uses post-step rendering. Added test definitions cover observed
medians with missing/zero times, missing-history compatibility, zero objectives,
fractional termination samples, output file names and CSV coverage counts.
No tests, benchmark modules, plotting functions or solver runs were executed.
Figure generation and visual appearance remain runtime-unverified.

## Incumbent trace verification (2026-11-05)

The benchmark opts into incumbent tracking; existing solver callers default to
disabled. MIPSOL callbacks record native discovery times and strict objective
improvements, respecting minimization/maximization. Final statistics reconstruct
the best-known objective at integer seconds from zero through native runtime,
plus a final fractional-second observation. No future value is backfilled into
earlier samples. Missing incumbents remain null, including time-to-first when
only a final native objective was available. Traces are persisted with finished
JSONL records; time-to-first is also a summary CSV column. No live disk stream or
additional bound/gap history was introduced.

Ruff lint/format, focused strict mypy on seven changed Python files, the 39-file
syntax/grid/solver-flow static checker and `git diff --check` passed. AST checks
cover benchmark opt-in, callback wiring, callback query codes, absence of callback
solver mutations and unchanged formulation call order. Added test definitions
cover both objective directions, non-improvements, subsecond and exact-second
events, zero objectives, unavailable incumbents and final-only observations.
These tests were not executed. Native callback delivery, timing and serialized
runtime results remain unverified; no tests, solves or campaigns were run.

## Workload calibration verification (2026-11-05)

User-approved defaults: reference load 0.75 for main and pilot; uniform deadlines
in [0.2, 3] seconds for all campaigns. Sensitivity retains loads 0.3 and 0.8.
For the same seven-flow reference pool, main/pilot arrival rates are 25% higher
than at load 0.6. Configuration, default-contract assertions, scenario fixtures
and README were updated together. The scenario tests obtain their load and
deadline bounds from configuration.

Ruff lint/format, focused strict mypy on the three changed Python files,
the 37-file compilation/grid/solver-flow static checker and `git diff --check`
passed. These are revised experimental assumptions, not measurements from a
pilot. No dynamic tests, profiling, exports, solves or campaigns were executed.
Run counts remain 770 main, 198 sensitivity and six pilot cases per objective.

## Pool expansion verification (2026-11-05)

Defaults now contain ten models and seven flows. Main enumerates 770 configurations
per objective; sensitivity samples flows 1/4/7 and models 1/5/10 for 198 runs.
The six pilot cases include the new maximum. Selecting all four objectives gives
3,080 main runs. Validation and subset bounds use configured/actual pool sizes,
including pilot checks, rather than the former hardcoded six/five limits.

For this change: Ruff lint/format passed; the static checker compiled 37 Python
files and verified YAML/TOML, the 770/198 grids, pool sizes and unchanged solver
call order; `git diff --check` passed. Targeted strict mypy passed on all seven
modified Python files with the documented external-library overrides.
An extended mypy scan of all 29 experiment Python files reports three pre-existing
`[index]` errors in unchanged `tests/test_solver.py:185-187`: nested indexing of
values typed as `object`. That file is unchanged relative to HEAD. The extended
scan is therefore not reported as clean.

No pytest, export, profiling, solve, pilot or plot was executed. The exporter now
accepts `--models` so the four added EfficientNets can be prepared explicitly
without replacing existing artifacts. Expanding the scenario changes pool-based
model selection, synthetic accuracies, arrival calibration and resulting run IDs.

## Framework verification record

* Ruff lint and formatting checks: passed for experiment source, scripts, tests
  and the four production files changed by this work.
* Python in-memory compilation: **32 files**. No module execution or `.pyc` writes.
* YAML and TOML parsing: passed. Static default-grid arithmetic: **770 main**,
  **198 sensitivity**, **11 ordered infrastructure configurations**, seed 42,
  300-second timeout, 0.01 gap, 60-second pilot, seven components,
  ten candidate models and seven flows. Counts are per objective; all four
  objectives give 3,080 main and 792 sensitivity runs, plus 24 pilot runs.
* AST comparison: native variable/constraint/objective/solution function call
  signatures and order inside `GurobiSolver._solve` match the repository HEAD.
* Git diff: queue, memory, network, assignment, variable, QoS and objective
  formulation modules unchanged. The problem builder only adds ACCURACY to the
  existing increasing-metric eligibility branch.
* Strict mypy: **32 files**, no issues, using existing root settings and an
  experiment source path. Targeted environmental overrides: matplotlib absent
  from the current environment; torchvision installed without typing metadata.
  Imports for those packages were excluded from missing-import diagnostics.
  Their runtime APIs are not proven by this type check. Existing root ONNX/ORT
  typing exclusions remain in force.
* Existing local torchvision source declares `get_model_builder` and
  `get_model_weights`; local ONNX source declares the external-data iterator used
  for artifact hashing. This is source inspection, not execution.
* `uv lock --project experiments/scalability --check --offline`: passed,
  **86 resolved packages** after selecting the explicit PyTorch CPU index.
  Static lock inspection confirms CPU-index sources for torch/torchvision and
  no NVIDIA, CUDA or Triton package entries.
* `git diff --check`: passed.

Static checker, run from the repository root:

```bash
python -B experiments/scalability/scripts/check_static.py
ruff check experiments/scalability \
  packages/shared/src/shared/model/model_variant.py \
  services/controller/src/controller/domain/optimization/solving/solver_config.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_problem_builder.py
ruff format --check experiments/scalability \
  packages/shared/src/shared/model/model_variant.py \
  services/controller/src/controller/domain/optimization/solving/solver_config.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_problem_builder.py
uv lock --project experiments/scalability --check --offline
git diff --check
```

For type checking in the existing environment without installing plotting/export
dependencies, the root configuration was copied to `/tmp/scalability-mypy.toml`,
`$MYPY_CONFIG_FILE_DIR` expanded to the absolute repository path, and these two
overrides appended:

```toml
[[tool.mypy.overrides]]
module = ["matplotlib", "matplotlib.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["torchvision", "torchvision.*"]
ignore_missing_imports = true
```

Then:

```bash
MYPYPATH=experiments/scalability/src mypy \
  --config-file=/tmp/scalability-mypy.toml \
  --no-incremental --cache-dir=/tmp/scalability-mypy --follow-imports=silent \
  experiments/scalability/src experiments/scalability/tests experiments/scalability/scripts \
  services/controller/src/controller/domain/optimization/solving/solver_config.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py \
  services/controller/src/controller/adapters/outbound/optimization/gurobi_problem_builder.py \
  packages/shared/src/shared/model/model_variant.py
```

This targeted check suppresses diagnostics in imported, unmodified production
modules; it is not a clean bill of health for the whole repository.

## Execution-path review

1. `plan` parses YAML and enumerates specs. It never discovers model files,
   profiles or calls `solve`.
2. Execution commands acquire a single-writer output lock before profiling or
   persisting anything. Local model discovery rejects missing artifacts before
   calling the profiler. Downloads exist only in the separate exporter and only
   for explicit weight mode `download`.
3. A full scenario is built once per seed. Model accuracy, model ordering,
   per-layer performance, worker identity, pairwise links, arrival weights and
   deadlines use separate addressed streams. DTO tensor sets are sorted before
   scenario hashing, preventing Python hash randomization from changing run IDs.
4. Profiling uses `DefaultModelProfiler`, its four production collaborators,
   `ArtifactWorkspace`, and existing DTO mappers. Cache identity covers artifact
   content, shape/accuracy and source/dependency versions. No Redis or S3 adapter
   is constructed.
5. Subset construction retains selected model/flow prefixes and the prescribed
   worker subsets. GPU times occupy CPU slots; GPU VRAM occupies solver RAM
   capacity. Device endpoints and inference-runner eligibility remain explicit.
6. Native model construction calls are unchanged. Observer invocation is in
   `finally` around `model.optimize()`, before native status handling, extraction,
   and model closure. Attributes requiring incumbents are guarded by SolCount;
   unavailable/infinite values become null. Optional seed/threads are set only
   when configured; existing callers retain their defaults.
7. Only native INFEASIBLE proves infeasibility. Native TIME_LIMIT branches on
   incumbent presence. Optimality classifications retain raw status and gap.
   Execution errors are persisted separately from mathematical infeasibility.
8. Started events precede solving; finished events follow extraction/classification.
   JSONL records are fsynced; CSV snapshots and manifests use replacement writes.
   Resume drops only a final uncommitted line, reruns incomplete/interrupted
   attempts, and skips completed timeouts/infeasibility. Committed corruption
   raises an error. The first started event embeds full scenario/software metadata.
9. `analyze` reads persisted events only. Timeout samples remain in runtime
   distributions. Counts expose missing runtimes/gaps and execution errors;
   no-incumbent runs contribute no invented objectives or gaps.

## Remaining runtime evidence

Tests are present but **not executed**. Real artifacts have not been exported or
profiled. Solver invocation, native parameter application, license availability,
status observations, recovery under actual termination, and matplotlib figure
generation remain runtime-unverified. The pilot and both campaigns have not run.
Peak-liveness is diagnostic; the authorized existing memory constraint remains.
