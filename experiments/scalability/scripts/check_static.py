"""Syntax, default-grid and production call-graph checks. Never imports benchmark code."""

import ast
import tomllib
from pathlib import Path

import yaml


def native_calls(source: str) -> list[str]:
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_solve"
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id in {"set_objective", "extract_solution"}
            )
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_formulation"
            )
        )
    ]
    names = []
    for node in sorted(calls, key=lambda node: node.lineno):
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        else:
            assert isinstance(node.func, ast.Attribute)
            names.append(node.func.attr)
    return names


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    experiment = root / "experiments/scalability"
    production = (
        "packages/shared/src/shared/model/model_variant.py",
        "services/controller/src/controller/domain/optimization/solving/solver_config.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_problem_builder.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_objective.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_qos_constraints.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_variables.py",
        "services/controller/src/controller/adapters/outbound/optimization/compact_bounds.py",
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
    progression = [[index, 0] for index in range(6)] + [
        [5, index] for index in range(1, 6)
    ]
    assert raw["infrastructure"] == progression
    assert raw["seeds"] == [42]
    for name, count in (("main", 330), ("sensitivity", 198)):
        campaign = raw[name]
        assert (
            len(raw["seeds"])
            * len(campaign["loads"])
            * len(campaign["flows"])
            * len(campaign["models"])
            * len(progression)
            == count
        )
    assert raw["main"]["loads"] == [0.6] and not raw["sensitivity"]["enabled"]
    assert raw["solver"]["timeout_s"] == 300 and raw["solver"]["relative_gap"] == 0.01
    assert (
        raw["pilot"]["timeout_s"] == 60 and raw["optimization"]["max_components"] == 7
    )
    assert len(raw["catalog"]["models"]) == 6 and raw["flows"]["pool_size"] == 5
    adapter = Path(
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py"
    )
    assert native_calls((root / adapter).read_text()) == [
        "prepare_problem",
        "add_variables",
        "add_assignment_constraints",
        "add_execution_constraints",
        "add_transmission_constraints",
        "add_execution_queue_constraints",
        "add_network_queue_constraints",
        "add_memory_constraints",
        "add_response_time_constraints",
        "set_objective",
        "extract_solution",
    ]
    optimizer = root / adapter.parent
    objective = ast.parse((optimizer / "gurobi_objective.py").read_text())
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
        if isinstance(node, ast.ClassDef) and node.name == "ObjectiveKind"
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
    }
    qos = ast.parse((optimizer / "gurobi_qos_constraints.py").read_text())
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
        "ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME",
        "ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY",
    }
    for name in ("compact_bounds.py", "gurobi_variables.py"):
        tree = ast.parse((optimizer / name).read_text())
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "max_mean_response_time"
            for node in ast.walk(tree)
        ), name
    print(
        f"Static inspection: {len(sources)} Python syntax trees; YAML/TOML valid; base grids 330/198; 11 infrastructures; four alternative objectives; conditional deadlines; no implicit deadline bounds; solver call order valid"
    )
    print(
        "No benchmark imports, tests, profiling, export, plotting or solver execution"
    )


if __name__ == "__main__":
    main()
