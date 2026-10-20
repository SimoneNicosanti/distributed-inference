from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from shared.model.keys import EdgeKey, LayerKey, TensorKey


class LayerDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    is_input: bool
    is_output: bool


class EdgeDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: LayerKey
    target: LayerKey
    tensors: frozenset[TensorKey] = Field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class TensorRoute:
    source: LayerKey
    destinations: tuple[LayerKey, ...]
    maximal_destinations: tuple[LayerKey, ...]


class _LayerAttributes(TypedDict):
    description: LayerDescription


class _EdgeAttributes(TypedDict):
    description: EdgeDescription


type _ModelVariantGraph = nx.DiGraph[
    LayerKey,
    _LayerAttributes,
    _EdgeAttributes,
]


class ModelVariantTopology:
    __slots__ = ("_graph",)

    def __init__(
        self,
        *,
        layers: Mapping[LayerKey, LayerDescription],
        edges: Iterable[EdgeDescription],
    ) -> None:
        edge_descriptions = tuple(edges)
        layer_keys = layers.keys()
        unknown_layer_keys = {
            layer_key
            for edge in edge_descriptions
            for layer_key in (edge.source, edge.target)
            if layer_key not in layer_keys
        }
        if unknown_layer_keys:
            raise ValueError(
                f"Edges reference unknown layers: {sorted(unknown_layer_keys)!r}"
            )

        edge_keys = [(edge.source, edge.target) for edge in edge_descriptions]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("Duplicate edges are not allowed")

        graph: _ModelVariantGraph = nx.DiGraph()
        for layer_key, description in layers.items():
            graph.add_node(layer_key, description=description)
        for edge_description in edge_descriptions:
            graph.add_edge(
                edge_description.source,
                edge_description.target,
                description=edge_description,
            )

        if not nx.is_directed_acyclic_graph(cast(Any, graph)):
            raise ValueError("Model variant topology must be a DAG")

        self._graph = cast(_ModelVariantGraph, nx.freeze(cast(Any, graph)))

    def layers_in_topological_order(self) -> tuple[LayerKey, ...]:
        return tuple(
            cast(
                Iterable[LayerKey],
                nx.topological_sort(cast(Any, self._graph)),
            )
        )

    def is_reachable(self, source: LayerKey, target: LayerKey) -> bool:
        """Return whether ``target`` is reachable from ``source`` in the DAG."""
        return bool(nx.has_path(cast(Any, self._graph), source, target))

    def tensor_routes(self) -> Mapping[TensorKey, TensorRoute]:
        sources: dict[TensorKey, LayerKey] = {}
        destinations: dict[TensorKey, set[LayerKey]] = {}
        for edge in self.edges().values():
            for tensor in edge.tensors:
                existing_source = sources.get(tensor)
                if existing_source is not None and existing_source != edge.source:
                    raise ValueError(f"Tensor {tensor!r} has multiple sources")
                sources[tensor] = edge.source
                destinations.setdefault(tensor, set()).add(edge.target)

        return {
            tensor: TensorRoute(
                source=sources[tensor],
                destinations=tuple(sorted(tensor_destinations)),
                maximal_destinations=tuple(
                    destination
                    for destination in sorted(tensor_destinations)
                    if not any(
                        destination != other and self.is_reachable(destination, other)
                        for other in tensor_destinations
                    )
                ),
            )
            for tensor, tensor_destinations in sorted(destinations.items())
        }

    def layers(self) -> Mapping[LayerKey, LayerDescription]:
        nodes = cast(
            Iterable[tuple[LayerKey, LayerDescription]],
            self._graph.nodes(data="description"),
        )
        return dict(nodes)

    def edges(self) -> Mapping[EdgeKey, EdgeDescription]:
        edges = cast(
            Iterable[tuple[LayerKey, LayerKey, EdgeDescription]],
            self._graph.edges(data="description"),
        )
        return {(source, target): description for source, target, description in edges}
