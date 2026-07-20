from __future__ import annotations


from pydantic import BaseModel, ConfigDict, PrivateAttr


from collections.abc import Mapping, Iterable
from typing import TypedDict, cast, Tuple, List

from enum import Enum
import networkx as nx


INPUT_LAYER_NAME = "InputLayer"
OUTPUT_LAYER_NAME = "OutputLayer"

AGGREGATED_LAYER_TYPE = "AggregatedLayer"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

    REGRESSION = "regression"


class LayerInfo(BaseModel):
    model_config = ConfigDict(frozen=False)

    name: str

    type: str

    flops: float
    weights_size: float

    inputs: dict[str, float]

    outputs: dict[str, float]

    is_input: bool
    is_output: bool

    is_aggregated: bool
    aggregated_layers: list[LayerInfo]


class EdgeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str

    tensors: dict[str, float]


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    accuracy: float
    task: TaskType


class ModelAttributes(TypedDict):
    info: ModelInfo


class LayerAttributes(TypedDict):
    info: LayerInfo


class EdgeAttributes(TypedDict):
    info: EdgeInfo


type RawModelGraph = nx.DiGraph[
    str,
    LayerAttributes,
    EdgeAttributes,
]
type LayerKey = str
type EdgeKey = Tuple[LayerKey, LayerKey]


def create_raw_model_graph() -> RawModelGraph:
    return nx.DiGraph()


class ModelGraph(BaseModel):
    model_config = ConfigDict(
        frozen=False,
    )

    info: ModelInfo | None = None

    _graph: RawModelGraph = PrivateAttr(
        default_factory=create_raw_model_graph,
    )

    def add_layer(self, layer_info: LayerInfo) -> None:
        self._graph.add_node(layer_info.name, info=layer_info)

    def remove_layer(self, layer: LayerKey) -> None:
        self._graph.remove_node(layer)

    def add_edge(self, edge_info: EdgeInfo) -> None:
        self._graph.add_edge(edge_info.source, edge_info.target, info=edge_info)

    def remove_edge(self, edge: EdgeKey) -> None:
        self._graph.remove_edge(edge[0], edge[1])

    def has_path(self, source: LayerKey, target: LayerKey) -> bool:
        return nx.has_path(cast(nx.DiGraph, self._graph), source, target)

    def get_all_layers(
        self,
    ) -> Mapping[LayerKey, LayerInfo]:
        return dict(
            cast(
                Iterable[tuple[LayerKey, LayerInfo]],
                self._graph.nodes(data="info"),
            )
        )

    def get_all_edges(
        self,
    ) -> Mapping[EdgeKey, EdgeInfo]:
        return cast(
            dict[EdgeKey, EdgeInfo],
            nx.get_edge_attributes(self._graph, "info"),  # type: ignore
        )

    def get_edge_info(self, edge: EdgeKey) -> EdgeInfo:
        return self._graph.get_edge_data(*edge)["info"]

    def get_layer_info(self, layer: LayerKey) -> LayerInfo:
        return self._graph.nodes[layer]["info"]

    def gat_layer_info_from_iterable(
        self, layers: Iterable[LayerKey]
    ) -> Mapping[LayerKey, LayerInfo]:
        return {layer: self.get_layer_info(layer) for layer in layers}

    def has_layer(self, layer: LayerKey) -> bool:
        return layer in self._graph.nodes

    def get_in_out_layer_degree(self, layer: LayerKey) -> Tuple[int, int]:
        return self._graph.in_degree(layer), self._graph.out_degree(layer)

    def get_layer_in_edges(self, layer: str) -> Mapping[EdgeKey, EdgeInfo]:
        in_edges_with_data: Iterable[tuple[LayerKey, LayerKey, EdgeInfo]] = (
            self._graph.in_edges(layer, data="info")
        )
        in_edges_dict = {(edge[0], edge[1]): edge[2] for edge in in_edges_with_data}
        return in_edges_dict

    def get_layer_out_edges(self, layer: LayerKey) -> Mapping[EdgeKey, EdgeInfo]:
        out_edges_with_data: Iterable[tuple[LayerKey, LayerKey, EdgeInfo]] = (
            self._graph.out_edges(layer, data="info")
        )
        out_edges_dict = {(edge[0], edge[1]): edge[2] for edge in out_edges_with_data}
        return out_edges_dict

    def get_reachable_from_layer(self, layer: str) -> Mapping[LayerKey, LayerInfo]:
        reachables = {layer} | nx.descendants(cast(nx.DiGraph, self._graph), layer)

        nodes_with_data = self._graph.nodes(data="info")
        selected_nodes = {
            name: info for name, info in nodes_with_data if name in reachables
        }

        return cast(Mapping[LayerKey, LayerInfo], selected_nodes)

    def find_tensor_producer(self, tensor_name: str) -> LayerKey | None:
        for layer, layer_info in self.get_all_layers().items():
            if tensor_name in layer_info.outputs:
                return layer
        return None

    def find_tensor_consumer_set(self, tensor_name: str) -> set[LayerKey]:
        consumers = set()
        for layer, layer_info in self.get_all_layers().items():
            if tensor_name in layer_info.inputs:
                consumers.add(layer)
        return consumers

    def remove_layers_from_iterable(self, layers: Iterable[LayerKey]) -> None:
        self._graph.remove_nodes_from(layers)

    def contract_edge_layers(self, edge: EdgeKey) -> None:

        if not self.is_edge_contractible(edge):
            raise ValueError(f"Edge {edge!r} cannot be contracted")

        source, target = edge

        source_info = self.get_layer_info(source)
        target_info = self.get_layer_info(target)

        contracted_nodes = {source, target}

        incoming, outgoing = self._get_layer_set_boundary(contracted_nodes)

        aggregated_name, aggregated_info = self._create_contracted_layer(
            source, target, source_info, target_info, incoming, outgoing
        )

        incoming_edges, outgoing_edges = self._create_contracted_edges(
            incoming, outgoing, aggregated_name
        )

        # Apply contraction
        self.remove_layers_from_iterable(contracted_nodes)
        self.add_layer(aggregated_info)

        for edge_info in incoming_edges:
            self.add_edge(edge_info)

        for edge_info in outgoing_edges:
            self.add_edge(edge_info)

    def _create_contracted_edges(
        self,
        incoming: dict[LayerKey, dict[str, float]],
        outgoing: dict[LayerKey, dict[str, float]],
        aggregated_name: LayerKey,
    ) -> Tuple[List[EdgeInfo], List[EdgeInfo]]:
        incoming_edges = [
            EdgeInfo(
                source=producer,
                target=aggregated_name,
                tensors=tensors,
            )
            for producer, tensors in incoming.items()
        ]

        outgoing_edges = [
            EdgeInfo(
                source=aggregated_name,
                target=consumer,
                tensors=tensors,
            )
            for consumer, tensors in outgoing.items()
        ]

        return incoming_edges, outgoing_edges

    def _create_contracted_layer(
        self,
        source: LayerKey,
        target: LayerKey,
        source_info: LayerInfo,
        target_info: LayerInfo,
        incoming: dict[LayerKey, dict[str, float]],
        outgoing: dict[LayerKey, dict[str, float]],
    ) -> Tuple[LayerKey, LayerInfo]:
        aggregated_inputs: dict[str, float] = {}
        for tensors in incoming.values():
            ModelGraph._merge_tensors(aggregated_inputs, tensors)

        aggregated_outputs: dict[str, float] = {}
        for tensors in outgoing.values():
            ModelGraph._merge_tensors(aggregated_outputs, tensors)

        aggregated_name = f"{source}∘{target}"

        if self.has_layer(aggregated_name):
            raise ValueError(f"Layer {aggregated_name!r} already exists")

        aggregated_layers = [
            *(
                source_info.aggregated_layers
                if source_info.is_aggregated
                else [source_info]
            ),
            *(
                target_info.aggregated_layers
                if target_info.is_aggregated
                else [target_info]
            ),
        ]

        aggregated_info = LayerInfo(
            name=aggregated_name,
            type=AGGREGATED_LAYER_TYPE,
            flops=source_info.flops + target_info.flops,
            weights_size=source_info.weights_size + target_info.weights_size,
            inputs=aggregated_inputs,
            outputs=aggregated_outputs,
            is_input=source_info.is_input or target_info.is_input,
            is_output=source_info.is_output or target_info.is_output,
            is_aggregated=True,
            aggregated_layers=aggregated_layers,
        )

        return aggregated_name, aggregated_info

    def is_edge_contractible(self, edge: EdgeKey) -> bool:

        if edge not in self.get_all_edges().keys():
            return False

        source, target = edge
        source_info = self.get_layer_info(source)
        target_info = self.get_layer_info(target)

        # The nodes connot be input or output
        if source_info.is_input or target_info.is_output:
            return False
        if target_info.is_input or source_info.is_output:
            return False

        # There is only one path between the two nodes
        graph_without_edge = nx.subgraph_view(
            self._graph,
            filter_edge=lambda u, v: (u, v) != edge,
        )
        if nx.has_path(cast(nx.DiGraph, graph_without_edge), source, target):
            return False

        return True

    @staticmethod
    def _merge_tensors(
        destination: dict[str, float],
        tensors: Mapping[str, float],
    ) -> None:
        for tensor_name, tensor_size in tensors.items():
            existing_size = destination.get(tensor_name)

            if existing_size is not None and existing_size != tensor_size:
                raise ValueError(
                    f"Inconsistent size for tensor {tensor_name!r}: "
                    f"{existing_size} != {tensor_size}"
                )

            destination[tensor_name] = tensor_size

    def _get_layer_set_boundary(
        self,
        layers: set[LayerKey],
    ) -> tuple[
        dict[LayerKey, dict[str, float]],
        dict[LayerKey, dict[str, float]],
    ]:
        incoming: dict[LayerKey, dict[str, float]] = {}
        outgoing: dict[LayerKey, dict[str, float]] = {}

        for layer in layers:
            # Edges: external node -> internal node
            for (
                source,
                _,
            ), edge_info in self.get_layer_in_edges(layer).items():
                if source in layers:
                    continue

                tensors = incoming.setdefault(source, {})
                ModelGraph._merge_tensors(tensors, edge_info.tensors)

            # Edges: internal node -> external node
            for (
                _,
                target,
            ), edge_info in self.get_layer_out_edges(layer).items():
                if target in layers:
                    continue

                tensors = outgoing.setdefault(target, {})
                ModelGraph._merge_tensors(tensors, edge_info.tensors)

        return incoming, outgoing

    def model_post_init(self, __context: object) -> None:
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        pass
        # graph: RawModelGraph = nx.DiGraph()

        # for layer in self.layers.values():
        #     graph.add_node(layer.name)

        # for edge in self.edges:
        #     if edge.source not in graph:
        #         raise ValueError(
        #             f"Unknown source layer: {edge.source!r}"
        #         )

        #     if edge.target not in graph:
        #         raise ValueError(
        #             f"Unknown target layer: {edge.target!r}"
        #         )

        #     graph.add_edge(edge.source, edge.target)

        # self._graph = graph
