from collections.abc import Iterator
from dataclasses import dataclass
from math import inf, isclose

import networkx as nx

from distributed_inference.domain.ModelGraphInfo import (
    ModelGraph,
    LayerKey,
    EdgeKey,
)

from pathlib import Path

from argparse import ArgumentParser, Namespace

pase_model_path = Path("/workspace/distributed-inference/assets/models")


@dataclass(frozen=True, slots=True)
class IdealCut:
    ideal: frozenset[str]
    cut_edges: tuple[EdgeKey, ...]
    total_tensor_size: float


@dataclass(frozen=True, slots=True)
class ExtremeIdealCuts:
    evaluated_ideals: int

    minimum_tensor_size: float
    minimum_cuts: tuple[IdealCut, ...]

    maximum_tensor_size: float
    maximum_cuts: tuple[IdealCut, ...]


def iter_ideal_cuts(
    model_graph: ModelGraph,
    *,
    include_trivial: bool = False,
) -> Iterator[IdealCut]:
    """Enumerate the ideals and their outgoing cuts.

    Args:
        model_graph:
            Model graph to analyse.

        include_trivial:
            Whether to include the empty ideal and the full ideal.
            Both have an empty cut with total tensor size equal to zero.

    Yields:
        Each ideal together with its outgoing cut and the total size
        of the tensors crossing that cut.
    """
    layers = model_graph.get_all_layers()
    edges = model_graph.get_all_edges()

    graph = nx.DiGraph()
    graph.add_nodes_from(layers)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The model graph must be a DAG.")

    edge_sizes: dict[EdgeKey, float] = {
        edge: float(
            sum(
                model_graph.get_default_sizes_for_tensors_set(
                    edge_info.tensors
                ).values()
            )
        )
        for edge, edge_info in edges.items()
    }

    # Principal ideal ↓v = {v} ∪ ancestors(v).
    principal_ideals: dict[str, frozenset[str]] = {
        layer: frozenset({layer, *nx.ancestors(graph, layer)}) for layer in graph.nodes
    }

    topological_order = list(nx.topological_sort(graph))
    number_of_layers = len(graph)

    for antichain in nx.antichains(
        graph,
        topo_order=topological_order,
    ):
        ideal_nodes: set[str] = set()

        # The antichain contains the maximal elements of the ideal.
        for maximal_layer in antichain:
            ideal_nodes.update(principal_ideals[maximal_layer])

        if not include_trivial and (
            not ideal_nodes or len(ideal_nodes) == number_of_layers
        ):
            continue

        cut_edges = tuple(
            sorted(
                (source, target)
                for source, target in graph.out_edges(ideal_nodes)
                if target not in ideal_nodes
            )
        )

        total_tensor_size = sum(edge_sizes[edge] for edge in cut_edges)

        yield IdealCut(
            ideal=frozenset(ideal_nodes),
            cut_edges=cut_edges,
            total_tensor_size=total_tensor_size,
        )


def find_extreme_ideal_cuts(
    model_graph: ModelGraph,
    *,
    include_trivial: bool = False,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
) -> ExtremeIdealCuts:
    """Find all ideals whose cut has minimum or maximum tensor size."""
    minimum_size = inf
    maximum_size = -inf

    minimum_cuts: list[IdealCut] = []
    maximum_cuts: list[IdealCut] = []

    evaluated_ideals = 0

    for ideal_cut in iter_ideal_cuts(
        model_graph,
        include_trivial=include_trivial,
    ):
        evaluated_ideals += 1
        size = ideal_cut.total_tensor_size

        if evaluated_ideals % 5000 == 0:
            print(f"Evaluated ideals: {evaluated_ideals}")

        if size < minimum_size and not isclose(
            size,
            minimum_size,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            minimum_size = size
            minimum_cuts = [ideal_cut]

        elif isclose(
            size,
            minimum_size,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            minimum_cuts.append(ideal_cut)

        if size > maximum_size and not isclose(
            size,
            maximum_size,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            maximum_size = size
            maximum_cuts = [ideal_cut]

        elif isclose(
            size,
            maximum_size,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            maximum_cuts.append(ideal_cut)

    if evaluated_ideals == 0:
        raise ValueError(
            "The graph has no non-trivial ideals. "
            "Set include_trivial=True to include the empty and full ideals."
        )

    return ExtremeIdealCuts(
        evaluated_ideals=evaluated_ideals,
        minimum_tensor_size=minimum_size,
        minimum_cuts=tuple(minimum_cuts),
        maximum_tensor_size=maximum_size,
        maximum_cuts=tuple(maximum_cuts),
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    args = parse_args()

    model_name = args.model
    model_path = pase_model_path / f"{model_name}.onnx"

    # model_proto = onnx.load(str(model_path))
    model_info = get_model_info(model_name)

    base_model_graph = profile_with_model_optimization(
        model_path, model_info, OptimizationLevel.BASIC
    )

    ext_model_graph = profile_with_model_optimization(
        model_path, model_info, OptimizationLevel.EXTENDED
    )

    agg_model_graph = compute_aggregate_model_graph(base_model_graph, ext_model_graph)

    pass


if __name__ == "__main__":
    main()
