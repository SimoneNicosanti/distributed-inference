from dataclasses import replace
from itertools import product

import pulp
import pytest

from model_manager.adapters.outbound.model_max_cut_size_computer.pulp_max_cut_size_computer import (
    PulpMaxCutSizeComputer,
)
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableTensorProperties,
)
from model_manager.domain.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)


def _compute(
    tensor_shapes: dict[str, InputShapeVariableTensorProperties] | None = None,
) -> float:
    if tensor_shapes is None:
        tensor_shapes = {"tensor": InputShapeVariableTensorProperties(shape=(262_144,))}
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
        {
            tensor: InputShapeInvariableTensorProperties(
                data_type="float32",
                data_type_size_in_bytes=4,
                shape_expression=properties.shape,
            )
            for tensor, properties in tensor_shapes.items()
        },
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


def _min_cut_instance(
    edges: tuple[tuple[str, str, str], ...],
    sizes_mib: dict[str, int],
    inputs: frozenset[str],
) -> tuple[ModelVariantTopology, dict[str, InputShapeVariableTensorProperties]]:
    topology = ModelVariantTopology()
    for layer in {layer for source, target, _ in edges for layer in (source, target)}:
        topology.add_layer(
            layer,
            LayerDescription(
                name=layer,
                type="Compute",
                is_input=layer in inputs,
                is_output=False,
            ),
        )
    for source, target, tensor in edges:
        topology.add_edge(
            (source, target),
            EdgeDescription(source=source, target=target, tensors=frozenset({tensor})),
        )
    return topology, {
        tensor: InputShapeVariableTensorProperties(shape=(size * 262_144,))
        for tensor, size in sizes_mib.items()
    }


@pytest.mark.parametrize(
    ("edges", "sizes_mib", "inputs", "expected"),
    (
        ((("input", "output", "t"),), {"t": 2}, frozenset({"input"}), 2),
        (
            (("input", "middle", "a"), ("middle", "output", "b")),
            {"a": 3, "b": 1},
            frozenset({"input"}),
            1,
        ),
        (
            (
                ("input", "left", "shared"),
                ("input", "right", "shared"),
                ("left", "output", "a"),
                ("right", "output", "b"),
            ),
            {"shared": 2, "a": 1, "b": 1},
            frozenset({"input"}),
            2,
        ),
        (
            (
                ("input_a", "merge", "a"),
                ("input_b", "merge", "b"),
                ("merge", "output", "c"),
            ),
            {"a": 2, "b": 3, "c": 10},
            frozenset({"input_a", "input_b"}),
            5,
        ),
    ),
)
def test_nontrivial_min_cut_matches_exhaustive_search(
    edges: tuple[tuple[str, str, str], ...],
    sizes_mib: dict[str, int],
    inputs: frozenset[str],
    expected: float,
) -> None:
    topology, shapes = _min_cut_instance(edges, sizes_mib, inputs)
    layers = tuple(topology.layers())
    cut_sizes = []
    for sides in product((False, True), repeat=len(layers)):
        left = {
            layer for layer, included in zip(layers, sides, strict=True) if included
        }
        if not inputs <= left or len(left) == len(layers):
            continue
        if any(target in left and source not in left for source, target, _ in edges):
            continue
        crossing = {
            tensor
            for source, target, tensor in edges
            if source in left and target not in left
        }
        cut_sizes.append(sum(sizes_mib[tensor] for tensor in crossing))

    assert min(cut_sizes) == expected
    assert PulpMaxCutSizeComputer.compute_model_variant_min_cut_size(
        topology, shapes
    ) == pytest.approx(expected)


def test_min_cut_falls_back_to_lp_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, shapes = _min_cut_instance(
        (("input", "middle", "a"), ("middle", "output", "b")),
        {"a": 1, "b": 3},
        frozenset({"input"}),
    )
    original_solve = pulp.LpProblem.solve
    modes: list[bool] = []

    def solve(problem: pulp.LpProblem, solver: pulp.HiGHS) -> pulp.LpSolveStats:
        modes.append(solver.mip)
        result = original_solve(problem, solver)
        if solver.mip:
            assert result.objective == pytest.approx(1)
            return replace(result, status=pulp.LpSolveStatus.TimeLimit)
        return result

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)

    assert PulpMaxCutSizeComputer.compute_model_variant_min_cut_size(
        topology, shapes
    ) == pytest.approx(0.5)
    assert modes == [True, False]


def test_min_cut_returns_zero_if_both_solves_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, shapes = _min_cut_instance(
        (("input", "output", "t"),), {"t": 2}, frozenset({"input"})
    )

    def solve(_problem: pulp.LpProblem, _solver: pulp.HiGHS) -> pulp.LpSolveStats:
        return pulp.LpSolveStats(status=pulp.LpSolveStatus.TimeLimit)

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)

    assert (
        PulpMaxCutSizeComputer.compute_model_variant_min_cut_size(topology, shapes)
        == 0.0
    )


@pytest.mark.parametrize("inputs", (frozenset(), frozenset({"a", "b"})))
def test_min_cut_rejects_invalid_boundary(inputs: frozenset[str]) -> None:
    topology, shapes = _min_cut_instance((("a", "b", "t"),), {"t": 1}, inputs)

    with pytest.raises(ValueError, match=r"input layer|non-input layer"):
        PulpMaxCutSizeComputer.compute_model_variant_min_cut_size(topology, shapes)


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
            "first": InputShapeVariableTensorProperties(shape=(262_144,)),
            "second": InputShapeVariableTensorProperties(shape=(131_072,)),
        }
    ) == pytest.approx(1.5)
    assert modes == [True, False]
