from __future__ import annotations


from pydantic import BaseModel, ConfigDict, PrivateAttr, Field


from collections.abc import Mapping, Iterable
from typing import TypedDict, cast, Tuple, List, Self

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


class ModelType(Enum):
    CNN = "cnn"
    VIT = "vit"
    BERT = "bert"


class FlopsInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    flops: dict[int, float] = Field(default_factory=dict)

    def __add__(self, other: object) -> Self:
        if not isinstance(other, FlopsInfo):
            return NotImplemented

        if not self.flops:
            return type(self)(flops=dict(other.flops))

        if not other.flops:
            return type(self)(flops=dict(self.flops))

        if self.flops.keys() != other.flops.keys():
            raise ValueError("FlopsInfo objects must have the same sequence lengths")

        return type(self)(
            flops={
                sequence_length: (value + other.flops[sequence_length])
                for sequence_length, value in self.flops.items()
            }
        )


class TensorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str

    shapes: dict[int, list[int]]  ## sequence length -> shape
    sizes: dict[int, float]  ## sequence length -> size


class LayerInfo(BaseModel):
    model_config = ConfigDict(frozen=False)

    name: str

    type: str

    flops: FlopsInfo
    weights_size: float

    inputs: set[str] = Field(default_factory=set)

    outputs: set[str] = Field(default_factory=set)

    is_input: bool
    is_output: bool

    is_aggregated: bool
    aggregated_layers: list[LayerInfo]


class EdgeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str

    tensors: set[str] = Field(default_factory=set)


class DynamicShapeType(Enum):
    BATCH = "batch"
    SEQUENCE = "sequence"


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str

    accuracy: float
    task: TaskType

    type: ModelType

    dynamic_shapes: dict[str, DynamicShapeType]  ## Shape-name to shape type

    batch_size: int = 1  ## Default to 1
    sequence_sizes: List[int] = [1]  ## Default to 1 for no sequence

    num_heads: int = 0
    hidden_size: int = 0


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

    model_info: ModelInfo | None = None
    _tensors_map: Mapping[str, TensorInfo] = PrivateAttr(default_factory=dict)

    _graph: RawModelGraph = PrivateAttr(
        default_factory=create_raw_model_graph,
    )

    def set_model_info(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def get_model_info(self) -> ModelInfo | None:
        return self.model_info

    def set_tensors_map(self, tensors_map: Mapping[str, TensorInfo]) -> None:
        self._tensors_map = tensors_map

    def get_tensors_map(self) -> Mapping[str, TensorInfo]:
        return self._tensors_map

    def get_default_sizes_for_tensors_set(
        self, tensors_set: set[str]
    ) -> dict[str, float]:
        assert self.model_info is not None
        min_size = min(self.model_info.sequence_sizes)

        return {
            tensor_name: self._tensors_map[tensor_name].sizes[min_size]
            for tensor_name in tensors_set
        }

    def get_default_flops_for_layer_set(self, layer_set: set[str]) -> dict[str, float]:
        assert self.model_info is not None
        min_size = min(self.model_info.sequence_sizes)

        return {
            layer_name: self._graph.nodes[layer_name]["info"].flops.flops[min_size]
            for layer_name in layer_set
        }

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

    def get_layer_info_from_iterable(
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

        incoming, outgoing = self._get_layer_set_boundary_tensors(contracted_nodes)

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

        if not nx.is_directed_acyclic_graph(self._graph):  # type: ignore
            raise ValueError("The graph must be a DAG.")

    def _create_contracted_edges(
        self,
        incoming: dict[LayerKey, set[str]],
        outgoing: dict[LayerKey, set[str]],
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
        incoming: dict[LayerKey, set[str]],
        outgoing: dict[LayerKey, set[str]],
    ) -> Tuple[LayerKey, LayerInfo]:
        aggregated_inputs: set[str] = set()
        for tensors in incoming.values():
            aggregated_inputs.update(tensors)

        aggregated_outputs: set[str] = set()
        for tensors in outgoing.values():
            aggregated_outputs.update(tensors)

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

    def extract_incoming_outgoing_tensors_of_sub_model(
        self, layers: set[LayerKey]
    ) -> Tuple[set[str], set[str]]:
        per_source_incoming, per_target_outgoing = self._get_layer_set_boundary_tensors(
            layers
        )

        incoming: set[str] = set()
        outgoing: set[str] = set()
        for _, source_incoming in per_source_incoming.items():
            incoming.update(source_incoming)

        for _, target_outgoing in per_target_outgoing.items():
            outgoing.update(target_outgoing)

        for _, layer_info in self.get_layer_info_from_iterable(layers).items():
            if layer_info.is_input:
                incoming.update(layer_info.outputs)
            if layer_info.is_output:
                outgoing.update(layer_info.outputs)

        return incoming, outgoing

    def get_topological_sort(self) -> List[LayerKey]:
        return list(nx.topological_sort(self._graph))  # type: ignore

    def _get_layer_set_boundary_tensors(
        self,
        layers: set[LayerKey],
    ) -> tuple[
        dict[LayerKey, set[str]],
        dict[LayerKey, set[str]],
    ]:
        incoming: dict[
            LayerKey, set[str]
        ] = {}  ## Map tensor source -> set of tensors received by layers in set
        outgoing: dict[
            LayerKey, set[str]
        ] = {}  ## Map tensor target -> set of tensors sent by layers in set

        for layer in layers:
            # Edges: external node -> internal node
            for (
                source,
                _,
            ), edge_info in self.get_layer_in_edges(layer).items():
                if source in layers:
                    continue

                tensors = incoming.setdefault(source, set())
                tensors.update(edge_info.tensors)

            # Edges: internal node -> external node
            for (
                _,
                target,
            ), edge_info in self.get_layer_out_edges(layer).items():
                if target in layers:
                    continue

                tensors = outgoing.setdefault(target, set())
                tensors.update(edge_info.tensors)

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
