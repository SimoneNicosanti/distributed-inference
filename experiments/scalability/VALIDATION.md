# Static-only validation

Implementation verification deliberately did **not** execute pytest (including
collection), benchmark modules, torchvision export, ONNX profiling, Gurobi solves,
pilot/campaign execution, plots, Docker or microservices. No environment sync or
package installation was performed. The experiment lockfile was resolved and
checked; this validates dependency resolution, not installation or runtime.

## Checks performed

* Ruff lint and formatting checks: passed for experiment source, scripts, tests
  and the four production files changed by this work.
* Python in-memory compilation: **32 files**. No module execution or `.pyc` writes.
* YAML and TOML parsing: passed. Static default-grid arithmetic: **330 main**,
  **198 sensitivity**, **11 ordered infrastructure configurations**, seed 42,
  300-second timeout, 0.01 gap, 60-second pilot, seven components.
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
