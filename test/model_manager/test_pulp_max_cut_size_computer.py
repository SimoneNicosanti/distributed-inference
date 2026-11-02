from dataclasses import replace

import pulp
import pytest

from model_manager.adapters.outbound.model_max_cut_size_computer.pulp_max_cut_size_computer import (
    PulpMaxCutSizeComputer,
)
from model_manager.domain.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from model_manager.domain.shape_profile import TensorShapeProperty


def _compute(
    tensor_shapes: dict[str, TensorShapeProperty] | None = None,
) -> float:
    if tensor_shapes is None:
        tensor_shapes = {"tensor": TensorShapeProperty(shape=(262_144,))}
    topology = ModelVariantTopology()
    for layer in ("source", "target"):
        topology.add_layer(
            layer,
            LayerDescription(
                name=layer, type="Compute", is_input=False, is_output=False
            ),
        )
    topology.add_edge(
        ("source", "target"),
        EdgeDescription(
            source="source", target="target", tensors=frozenset(tensor_shapes)
        ),
    )
    return PulpMaxCutSizeComputer.compute_model_variant_max_cut_size(
        topology,
        tensor_shapes,
    )


def test_returns_integer_optimum() -> None:
    assert _compute() == pytest.approx(1.0)


def test_falls_back_to_lp_optimum_instead_of_integer_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve = pulp.LpProblem.solve
    modes: list[bool] = []

    def solve(problem: pulp.LpProblem, solver: pulp.HiGHS) -> pulp.LpSolveStats:
        modes.append(solver.mip)
        if len(modes) == 1:
            # Give the test instance an integrality gap: MIP=0, LP=0.5.
            problem += problem.objective <= 0.5
        result = original_solve(problem, solver)
        if len(modes) == 1:
            assert result.objective == pytest.approx(0.0)
            return replace(result, status=pulp.LpSolveStatus.TimeLimit)
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)

    assert _compute() == pytest.approx(0.5)
    assert modes == [True, False]


@pytest.mark.parametrize("failure", ("status", "missing", "nonfinite", "exception"))
def test_returns_sum_of_all_tensor_sizes_when_both_solves_fail(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    modes: list[bool] = []

    def solve(_problem: pulp.LpProblem, _solver: pulp.HiGHS) -> pulp.LpSolveStats:
        modes.append(_solver.mip)
        if failure == "exception":
            raise pulp.PulpSolverError("Solver failed")
        if failure == "status":
            return pulp.LpSolveStats(status=pulp.LpSolveStatus.TimeLimit)
        return pulp.LpSolveStats(
            status=pulp.LpSolveStatus.Optimal,
            objective=None if failure == "missing" else float("nan"),
        )

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)

    assert _compute(
        {
            "first": TensorShapeProperty(shape=(262_144,)),
            "second": TensorShapeProperty(shape=(131_072,)),
        }
    ) == pytest.approx(1.5)
    assert modes == [True, False]
