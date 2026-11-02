"""Syntax, default-grid and production call-graph checks. Never imports benchmark code."""

import ast
import subprocess
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
        and isinstance(node.func, ast.Name)
        and (
            node.func.id.startswith("add_")
            or node.func.id in {"set_objective", "extract_solution"}
        )
    ]
    return [ast.dump(node) for node in sorted(calls, key=lambda node: node.lineno)]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    experiment = root / "experiments/scalability"
    production = (
        "packages/shared/src/shared/model/model_variant.py",
        "services/controller/src/controller/domain/optimization/solving/solver_config.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_problem_builder.py",
        "services/controller/src/controller/adapters/outbound/optimization/gurobi_solver.py",
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
    baseline = subprocess.check_output(
        ["git", "show", f"HEAD:{adapter.as_posix()}"], cwd=root, text=True
    )
    assert native_calls(baseline) == native_calls((root / adapter).read_text()), (
        "Production formulation call signatures or ordering changed"
    )
    print(
        f"Static inspection: {len(sources)} Python syntax trees; YAML/TOML valid; grids 330/198; 11 infrastructures; native formulation calls unchanged"
    )
    print(
        "No benchmark imports, tests, profiling, export, plotting or solver execution"
    )


if __name__ == "__main__":
    main()
