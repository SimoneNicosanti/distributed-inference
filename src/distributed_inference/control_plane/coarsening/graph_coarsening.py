import copy
from typing import List, Tuple

import numpy as np

from distributed_inference.domain.model_graph_info import (
    EdgeKey,
    ModelGraph,
)


def coarse_graph(model_graph: ModelGraph, iterations: int) -> ModelGraph:
    coarse_model_graph = copy.deepcopy(model_graph)
    for it in range(iterations):
        print(f"Iteration {it}")
        sorted_score_list = compute_sorted_score_list(coarse_model_graph)
        applied = False
        for edge, score in sorted_score_list:
            if (
                score > 0
                and coarse_model_graph.is_edge_contractible(edge)
                and is_linear_edge(coarse_model_graph, edge)
            ):
                print(f"Contracting edge {edge} with score {score}")
                coarse_model_graph.contract_edge_layers(edge)
                applied = True
                break
        if not applied:
            break
    return coarse_model_graph


def is_linear_edge(model_graph: ModelGraph, edge: EdgeKey) -> bool:
    src_out_degree = len(model_graph.get_layer_out_edges(edge[0]))
    dst_in_degree = len(model_graph.get_layer_in_edges(edge[1]))

    return src_out_degree == 1 or dst_in_degree == 1


def compute_sorted_score_list(model_graph: ModelGraph) -> List[Tuple[EdgeKey, float]]:
    ## TODO: Validate this kind of scoring or replace it with some other heuristic
    flops_dict = {
        layer: layer_info.flops
        for layer, layer_info in model_graph.get_all_layers().items()
    }
    edge_sizes_dict = {
        edge: sum(edge_info.tensors.values())
        for edge, edge_info in model_graph.get_all_edges().items()
    }

    log_flops = {layer: np.log(1 + flops) for layer, flops in flops_dict.items()}
    log_flops_scale = max(float(np.median(list(log_flops.values()))), 1e-6)
    normalized_flops_dict = {
        layer: log_flops[layer] / log_flops_scale for layer in flops_dict.keys()
    }

    log_edge_sizes = {
        edge: np.log(1 + edge_size) for edge, edge_size in edge_sizes_dict.items()
    }
    log_edge_sizes_scale = max(float(np.median(list(log_edge_sizes.values()))), 1e-6)
    normalized_edge_sizes_dict = {
        edge: log_edge_sizes[edge] / log_edge_sizes_scale
        for edge in edge_sizes_dict.keys()
    }

    edge_scores: dict[EdgeKey, float] = {}
    for edge in edge_sizes_dict.keys():
        edge_src, edge_dst = edge

        score_num = normalized_edge_sizes_dict[edge]
        score_den = (
            1e-6 + normalized_flops_dict[edge_src] + normalized_flops_dict[edge_dst]
        )

        edge_scores[edge] = float(score_num / score_den)

    score_list = list(edge_scores.items())
    sorted_score_list = sorted(score_list, key=lambda elem: elem[1], reverse=True)

    return sorted_score_list
