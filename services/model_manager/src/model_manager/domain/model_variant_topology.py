from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypedDict, cast

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from shared.model.keys import EdgeKey, LayerKey, TensorKey

INPUT_LAYER_NAME: LayerKey = "InputLayer"
OUTPUT_LAYER_NAME: LayerKey = "OutputLayer"

LAYER_DESCRIPTION_PARAM_NAME = "description"
EDGE_DESCRIPTION_PARAM_NAME = "description"


class LayerDescription(BaseModel):
    model_config = ConfigDict(frozen=False)

    name: str
    type: str

    is_input: bool
    is_output: bool


class EdgeDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: LayerKey
    target: LayerKey

    tensors: set[TensorKey] = Field(default_factory=set)


class LayerAttributes(TypedDict):
    description: LayerDescription


class EdgeAttributes(TypedDict):
    description: EdgeDescription


type RawModelGraph = nx.DiGraph[
    str,
    LayerAttributes,
    EdgeAttributes,
]


## TODO: Implement serialization logic for this class
class ModelVariantTopology(BaseModel):
    model_config = ConfigDict(
        frozen=False,
    )

    @staticmethod
    def create_raw_model_graph() -> RawModelGraph:
        return nx.DiGraph()

    _graph: RawModelGraph = PrivateAttr(
        default_factory=create_raw_model_graph,
    )

    def add_layer(self, layer_key: LayerKey, layer_descrip: LayerDescription) -> None:
        additional_dict = {LAYER_DESCRIPTION_PARAM_NAME: layer_descrip}
        self._graph.add_node(layer_key, **additional_dict)

    def add_edge(self, edge_key: EdgeKey, edge_descript: EdgeDescription) -> None:
        additional_dict = {EDGE_DESCRIPTION_PARAM_NAME: edge_descript}
        self._graph.add_edge(edge_key[0], edge_key[1], **additional_dict)

    def has_path(self, source: LayerKey, target: LayerKey) -> bool:
        graph = cast(Any, self._graph)
        return nx.has_path(graph, source, target)

    def layers(
        self,
    ) -> dict[LayerKey, LayerDescription]:

        return dict(
            cast(
                Iterable[tuple[LayerKey, LayerDescription]],
                self._graph.nodes(data=LAYER_DESCRIPTION_PARAM_NAME),
            )
        )

    def edges(
        self,
    ) -> Mapping[EdgeKey, EdgeDescription]:
        edges_with_data: Iterable[tuple[LayerKey, LayerKey, EdgeDescription]] = (
            self._graph.edges(data=EDGE_DESCRIPTION_PARAM_NAME)
        )
        return {
            (source, target): description
            for source, target, description in edges_with_data
        }

    def get_edge_description(self, edge: EdgeKey) -> EdgeDescription:
        attributes = self._graph.get_edge_data(*edge)
        if attributes is None:
            raise KeyError(f"Unknown edge: {edge!r}")
        return attributes["description"]

    def get_layer_description(self, layer: LayerKey) -> LayerDescription:
        return self._graph.nodes[layer]["description"]

    def get_layer_info_from_iterable(
        self, layers: Iterable[LayerKey]
    ) -> Mapping[LayerKey, LayerDescription]:
        return {layer: self.get_layer_description(layer) for layer in layers}

    def has_layer(self, layer: LayerKey) -> bool:
        return layer in self._graph.nodes

    def get_in_out_layer_degree(self, layer: LayerKey) -> tuple[int, int]:
        return self._graph.in_degree(layer), self._graph.out_degree(layer)

    def get_layer_in_edges(self, layer: LayerKey) -> Mapping[EdgeKey, EdgeDescription]:
        in_edges_with_data: Iterable[tuple[LayerKey, LayerKey, EdgeDescription]] = (
            self._graph.in_edges(layer, data=EDGE_DESCRIPTION_PARAM_NAME)
        )
        in_edges_dict = {(edge[0], edge[1]): edge[2] for edge in in_edges_with_data}
        return in_edges_dict

    def get_layer_out_edges(self, layer: LayerKey) -> Mapping[EdgeKey, EdgeDescription]:
        out_edges_with_data: Iterable[tuple[LayerKey, LayerKey, EdgeDescription]] = (
            self._graph.out_edges(layer, data=EDGE_DESCRIPTION_PARAM_NAME)
        )
        out_edges_dict = {(edge[0], edge[1]): edge[2] for edge in out_edges_with_data}
        return out_edges_dict

    def get_reachable_from_layer(
        self, layer: LayerKey
    ) -> Mapping[LayerKey, LayerDescription]:
        graph = cast(Any, self._graph)
        reachables = {layer} | nx.descendants(graph, layer)

        nodes_with_data = self._graph.nodes(data=LAYER_DESCRIPTION_PARAM_NAME)
        selected_nodes = {
            name: info for name, info in nodes_with_data if name in reachables
        }

        return cast(Mapping[LayerKey, LayerDescription], selected_nodes)

    def find_tensor_producer(self, tensor_key: TensorKey) -> LayerKey | None:
        producers = {
            source
            for (source, _), description in self.edges().items()
            if tensor_key in description.tensors
        }
        if len(producers) > 1:
            raise ValueError(f"Tensor {tensor_key!r} has multiple producers")
        return next(iter(producers), None)

    def find_tensor_consumer_set(self, tensor_key: TensorKey) -> set[LayerKey]:
        return {
            target
            for (_, target), description in self.edges().items()
            if tensor_key in description.tensors
        }

    def remove_layers_from_iterable(self, layers: Iterable[LayerKey]) -> None:
        self._graph.remove_nodes_from(layers)

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self._graph)  # type: ignore

    def extract_incoming_outgoing_tensors_of_partition(
        self, layers: set[LayerKey]
    ) -> tuple[set[TensorKey], set[TensorKey]]:
        unknown_layers = layers - self.layers().keys()
        if unknown_layers:
            raise ValueError(f"Unknown layers: {sorted(unknown_layers)!r}")

        per_source_incoming, per_target_outgoing = self._get_layer_set_boundary_tensors(
            layers
        )

        incoming: set[TensorKey] = set()
        outgoing: set[TensorKey] = set()
        for _, source_incoming in per_source_incoming.items():
            incoming.update(source_incoming)

        for _, target_outgoing in per_target_outgoing.items():
            outgoing.update(target_outgoing)

        return incoming, outgoing

    def get_topological_sort(self) -> list[LayerKey]:
        return list(nx.topological_sort(self._graph))  # type: ignore

    def _get_layer_set_boundary_tensors(
        self,
        layers: set[LayerKey],
    ) -> tuple[
        dict[LayerKey, set[TensorKey]],
        dict[LayerKey, set[TensorKey]],
    ]:
        incoming: dict[
            LayerKey, set[TensorKey]
        ] = {}  ## Map tensor source -> set of tensors received by layers in set
        outgoing: dict[
            LayerKey, set[TensorKey]
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

    ## TODO: Add serialization and deserialization
