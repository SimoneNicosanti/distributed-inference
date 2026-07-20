from __future__ import annotations


from networkx.drawing.nx_pydot import to_pydot
import matplotlib.pyplot as plt
from distributed_inference.control_plane.coarsening.graph_coarsening import (
    compute_sorted_score_list,
)

from distributed_inference.model_management.model_profile import profile_model
from distributed_inference.model_management.model_profile_agg import (
    profile_model_with_aggregation,
)
from distributed_inference.control_plane.coarsening.graph_coarsening import (
    compute_sorted_score_list,
    coarse_graph,
)
import onnx
import networkx as nx
import onnxruntime as ort
from distributed_inference.domain.ModelGraphInfo import LayerInfo, ModelGraph, EdgeKey

from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")

MODEL_NAME = "yolo11x"

onnx_model = onnx.load(f"./../../../assets/models/{MODEL_NAME}.onnx")
agg_model_graph = profile_model_with_aggregation(onnx_model)

# base_model_graph = profile_model(onnx_model)

print("Numero di nodi Base: " + str(len(agg_model_graph.get_all_layers())))
print("Numero di archi Base: " + str(len(agg_model_graph.get_all_edges())))

# scores = compute_score_dict(agg_model_graph)
# sorted_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
# print(sorted_list)

coarse_model_graph = coarse_graph(agg_model_graph, 600)

print("Numero di nodi Coarse: " + str(len(coarse_model_graph.get_all_layers())))
print("Numero di archi Coarse: " + str(len(coarse_model_graph.get_all_edges())))

# for layer, layer_info in coarse_model_graph.get_all_layers().items():
#     if layer_info.is_aggregated:
#         print(layer_info)
#         print()

# for edge, edge_info in coarse_model_graph.get_all_edges().items():
#     print(edge_info)
# filter(lambda x: x[0])

# for layer, layer_info in agg_model_graph.get_all_layers().items():
#     print(layer_info)
#     print()


from collections.abc import Iterator
from dataclasses import dataclass
from math import inf, isclose

import networkx as nx


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
        edge: float(sum(edge_info.tensors.values()))
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


for iterations in range(50, 451, 50):
    coarse_model_graph = coarse_graph(agg_model_graph, iterations)

    analysis = find_extreme_ideal_cuts(coarse_model_graph)
    print("Iterazioni:", iterations)
    print("\tNumero di nodi Coarse:", len(coarse_model_graph.get_all_layers()))
    print("\tNumero di archi Coarse:", len(coarse_model_graph.get_all_edges()))
    print("\tIdeals analizzati:", analysis.evaluated_ideals)

    print("\tCut minimo:", analysis.minimum_tensor_size)
    # for result in analysis.minimum_cuts:
    #     print("Ideal:", sorted(result.ideal))
    #     print("Archi:", result.cut_edges)

    print("\tCut massimo:", analysis.maximum_tensor_size)
# for result in analysis.maximum_cuts:
#     print("Ideal:", sorted(result.ideal))
#     print("Archi:", result.cut_edges)
