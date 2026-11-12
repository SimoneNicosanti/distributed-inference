"""Syntax, default-grid and production call-graph checks. Never imports benchmark code."""

import ast
import tomllib
from pathlib import Path

import yaml


def native_calls(source: str, method_name: str) -> list[str]:
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    calls = sorted(
        (
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    return [ast.unparse(node.func) for node in calls]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    experiment = root / "experiments/scalability"
    production = (
        "packages/shared/src/shared/model/model_variant.py",
        "services/controller/src/controller/domain/optimization/solving/formulation/compact/compact_objective_kind.py",
        "services/controller/src/controller/application/optimization/formulation/compact/compact_formulation_data_builder.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/solver/gurobi_solver.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/model_builder/formulations/compact/collaborators/objective_builder.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/model_builder/formulations/compact/collaborators/response_time/constraints.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/model_builder/formulations/compact/compact_variables.py",
        "services/controller/src/controller/application/optimization/formulation/compact/bounds.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/trackers/gurobi_incumbent_tracker.py",
    )
    sources = sorted(
        path
        for directory in ("src", "scripts", "tests")
        for path in (experiment / directory).rglob("*.py")
    ) + [root / path for path in production]
    for path in sources:
        compile(path.read_text(), str(path), "exec")
    with (experiment / "pyproject.toml").open("rb") as handle:
        tomllib.load(handle)
    with (experiment / "uv.lock").open("rb") as handle:
        tomllib.load(handle)
    raw = yaml.safe_load((experiment / "config/default.yaml").read_text())
    progression = [[index, 0] for index in range(11)]
    assert raw["infrastructure"] == progression
    assert raw["seeds"] == [42]
    for name, count in (("main", 770), ("sensitivity", 198)):
        campaign = raw[name]
        assert (
            len(raw["seeds"])
            * len(campaign["loads"])
            * len(campaign["flows"])
            * len(campaign["models"])
            * len(progression)
            == count
        )
    assert raw["main"]["loads"] == [0.7] and not raw["sensitivity"]["enabled"]
    assert raw["pilot"]["load"] == 0.7
    assert raw["flows"]["deadline_s"] == [0.2, 3.0]
    assert raw["sensitivity"]["loads"] == [0.3, 0.8]
    assert raw["solver"]["timeout_s"] == 300 and raw["solver"]["relative_gap"] == 0.01
    assert (
        raw["pilot"]["timeout_s"] == 60 and raw["optimization"]["max_components"] == 7
    )
    assert len(raw["catalog"]["models"]) == 10 and raw["flows"]["pool_size"] == 7
    assert raw["main"]["models"] == list(range(1, 11))
    assert raw["main"]["flows"] == list(range(1, 8))
    assert raw["sensitivity"]["models"] == [1, 5, 10]
    assert raw["sensitivity"]["flows"] == [1, 4, 7]
    assert len(raw["pilot"]["cases"]) == 6
    assert any(
        case["flows"] == 7 and case["models"] == 10 for case in raw["pilot"]["cases"]
    )
    assert all(
        1 <= case["flows"] <= 7 and 1 <= case["models"] <= 10
        for case in raw["pilot"]["cases"]
    )
    adapter = Path(
        "services/controller/src/controller/adapters/outbound/optimization/gurobi/solver/gurobi_solver.py"
    )
    optimizer = (
        root / "services/controller/src/controller/adapters/outbound/optimization"
    )
    full_strategy = (
        root
        / "services/controller/src/controller/application/optimization/strategies/full_solver_strategy.py"
    )
    assert native_calls(full_strategy.read_text(), "solve") == [
        "self._formulation_data_builder.build_formulation_data",
        "self._formulation_model_builder.build_formulation_model",
        "self._solver_config_builder.build_solver_config",
        "self._solver.solve_formulation_model",
        "self._solved_model_translator.translate_solved_formulation_model",
    ]
    full_tree = ast.parse(full_strategy.read_text())
    solve = next(
        node
        for node in ast.walk(full_tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "solve"
    )
    assert [arg.arg for arg in solve.args.args] == [
        "self",
        "requests_by_flow",
        "context",
    ]
    scope = next(node for node in solve.body if isinstance(node, ast.AsyncWith))
    assert any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "translate_solved_formulation_model"
        for node in ast.walk(scope)
    )
    compiler = (
        optimizer
        / "gurobi/model_builder/formulations/compact/gurobi_compact_formulation_compiler.py"
    )
    assert native_calls(compiler.read_text(), "compile_formulation") == [
        "self._create_variables",
        "self._assignment.add_constraints",
        "self._execution.add_constraints",
        "self._compute_queue.add_constraints",
        "self._network.add_constraints",
        "self._memory.add_constraints",
        "self._response_time.add_constraints",
        "self._objective.set_objective",
    ]
    builder = optimizer / "gurobi/model_builder/gurobi_formulation_model_builder.py"
    for path in (full_strategy, compiler, builder):
        compile(path.read_text(), str(path), "exec")
    solver_tree = ast.parse((root / adapter).read_text())
    optimize_calls = [
        node
        for node in ast.walk(solver_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "optimize"
    ]
    assert len(optimize_calls) == 1 and len(optimize_calls[0].args) == 1
    assert isinstance(optimize_calls[0].args[0], ast.Name)
    assert optimize_calls[0].args[0].id == "callback"
    tracker = ast.parse(
        (optimizer / "gurobi/trackers/gurobi_incumbent_tracker.py").read_text()
    )
    callback = next(
        node
        for node in ast.walk(tracker)
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert {
        ast.unparse(node.args[0])
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cbGet"
    } == {"GRB.Callback.RUNTIME", "GRB.Callback.MIPSOL_OBJ"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"terminate", "cbCut", "cbLazy", "cbSetSolution"}
        for node in ast.walk(tracker)
    )
    runner = ast.parse((experiment / "src/scalability_benchmark/solver.py").read_text())
    assert any(
        isinstance(node, ast.keyword)
        and node.arg == "record_incumbent_history"
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in ast.walk(runner)
    )
    strategy_builder = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_solver_strategy"
    )
    configured_strategy = next(
        node
        for node in ast.walk(strategy_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FullSolverStrategy"
    )
    assert {keyword.arg for keyword in configured_strategy.keywords} == {
        "formulation_data_builder",
        "formulation_model_builder",
        "solver_config_builder",
        "solver",
        "solved_model_translator",
        "formulation_config",
    }
    recorder = next(
        node
        for node in runner.body
        if isinstance(node, ast.ClassDef) and node.name == "_RecordingSolver"
    )
    assert any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Await)
        and isinstance(node.value.value, ast.Call)
        and isinstance(node.value.value.func, ast.Attribute)
        and node.value.value.func.attr == "solve_formulation_model"
        for node in ast.walk(recorder)
    )
    configuration_tree = ast.parse(
        (experiment / "src/scalability_benchmark/configuration.py").read_text()
    )
    objectives = next(
        node.value
        for node in configuration_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BENCHMARK_OBJECTIVES"
            for target in node.targets
        )
    )
    assert isinstance(objectives, ast.Tuple) and len(objectives.elts) == 4
    assert all(
        isinstance(item, ast.Attribute) and item.attr != "CONSTANT"
        for item in objectives.elts
    )
    execution_profiles = ast.parse(
        (experiment / "src/scalability_benchmark/profiling/execution.py").read_text()
    )
    backend_profile = next(
        node
        for node in ast.walk(execution_profiles)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "BackendExecutionProfile"
    )
    assert ast.unparse(backend_profile.args[0]) == "Backend(hardware.backend)"
    hardware = ast.parse(
        (experiment / "src/scalability_benchmark/hardware.py").read_text()
    )
    worker_record = next(
        node.value
        for node in ast.walk(hardware)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    worker_metadata = {
        key.value: ast.unparse(value)
        for key, value in zip(worker_record.keys, worker_record.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    assert (
        worker_metadata["solver_backend_slot"]
        == "self.resource.execution_backend.value"
    )
    assert (
        worker_metadata["solver_memory_capacity_bytes"]
        == "self.resource.available_inference_memory_bytes"
    )
    analysis = ast.parse(
        (experiment / "src/scalability_benchmark/analysis.py").read_text()
    )
    analyze = next(
        node
        for node in analysis.body
        if isinstance(node, ast.FunctionDef) and node.name == "analyze"
    )
    analysis_calls = {
        node.func.id
        for node in ast.walk(analyze)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "read_journal",
        "plot_runtime_progression",
        "plot_aggregate_times",
        "plot_first_incumbent",
    } <= analysis_calls
    assert not {"execute_solver", "GurobiSolver"} & analysis_calls
    assert "plot_incumbent_history" not in analysis_calls
    aggregate_plots = ast.parse(
        (experiment / "src/scalability_benchmark/aggregate_time_plots.py").read_text()
    )
    assert {
        node.func.attr
        for node in ast.walk(aggregate_plots)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } >= {"boxplot", "step", "imshow", "savefig"}
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and any(
            keyword.arg == "where"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "post"
            for keyword in node.keywords
        )
        for node in ast.walk(analysis)
    )
    objective = ast.parse(
        (
            optimizer
            / "gurobi/model_builder/formulations/compact/collaborators/objective_builder.py"
        ).read_text()
    )
    calls = [
        node.func.attr
        for node in ast.walk(objective)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.count("setObjective") == 1 and "setObjectiveN" not in calls
    assert calls.count("addGenConstrIndicator") == 1
    configuration = ast.parse((root / production[1]).read_text())
    kinds = next(
        node
        for node in configuration.body
        if isinstance(node, ast.ClassDef) and node.name == "CompactObjectiveKind"
    )
    assert {
        target.id
        for node in kinds.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    } == {
        "ARRIVAL_WEIGHTED_RESPONSE_TIME",
        "ARRIVAL_WEIGHTED_ACCURACY",
        "DEADLINE_VIOLATIONS",
        "ARRIVAL_WEIGHTED_NORMALIZED_DELAY",
        "CONSTANT",
    }
    qos = ast.parse(
        (
            optimizer
            / "gurobi/model_builder/formulations/compact/collaborators/response_time/constraints.py"
        ).read_text()
    )
    guards = [
        node
        for node in ast.walk(qos)
        if isinstance(node, ast.If) and "objective_kind" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    test = guards[0].test
    assert isinstance(test, ast.Compare) and isinstance(test.ops[0], ast.In)
    assert isinstance(test.comparators[0], ast.Set)
    assert {ast.unparse(item) for item in test.comparators[0].elts} == {
        "CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME",
        "CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY",
    }
    for path in (
        root
        / "services/controller/src/controller/application/optimization/formulation/compact/bounds.py",
        optimizer / "gurobi/model_builder/formulations/compact/compact_variables.py",
    ):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "max_mean_response_time"
            for node in ast.walk(tree)
        ), path.name
    print(
        f"Static inspection: {len(sources) + 3} Python syntax trees; YAML/TOML valid; base grids 770/198; 10 models; 7 flows; 11 infrastructures; four alternative objectives; conditional deadlines; no implicit deadline bounds; solver call order, incumbent callback and offline plotting wiring valid"
    )
    print(
        "No benchmark imports, tests, profiling, export, plotting or solver execution"
    )


if __name__ == "__main__":
    main()
