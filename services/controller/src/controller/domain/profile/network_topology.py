import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import networkx as nx

from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class NetworkConnection:
    source: WorkerId
    target: WorkerId
    throughput_bytes_per_sec: float
    round_trip_time_s: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.throughput_bytes_per_sec)
            or self.throughput_bytes_per_sec <= 0
        ):
            raise ValueError("Throughput must be finite and positive")
        if not math.isfinite(self.round_trip_time_s) or self.round_trip_time_s < 0:
            raise ValueError("Round-trip time must be finite and non-negative")


class _WorkerAttributes(TypedDict):
    pass


class _ConnectionAttributes(TypedDict):
    connection: NetworkConnection


type NetworkConnectionKey = tuple[WorkerId, WorkerId]
type _NetworkGraph = nx.DiGraph[
    WorkerId,
    _WorkerAttributes,
    _ConnectionAttributes,
]


class NetworkTopology:
    __slots__ = ("_graph",)

    def __init__(
        self,
        *,
        workers: Iterable[WorkerId],
        connections: Iterable[NetworkConnection],
    ) -> None:
        worker_ids = frozenset(workers)
        network_connections = tuple(connections)

        unknown_worker_ids = {
            worker_id
            for connection in network_connections
            for worker_id in (connection.source, connection.target)
            if worker_id not in worker_ids
        }
        if unknown_worker_ids:
            unknown_ids = sorted(
                str(worker_id.service_id) for worker_id in unknown_worker_ids
            )
            raise ValueError(f"Connections reference unknown workers: {unknown_ids!r}")

        if any(
            connection.source == connection.target for connection in network_connections
        ):
            raise ValueError("Self-connections are not allowed")

        connection_keys = [
            (connection.source, connection.target) for connection in network_connections
        ]
        if len(connection_keys) != len(set(connection_keys)):
            raise ValueError("Duplicate directed connections are not allowed")

        graph: _NetworkGraph = nx.DiGraph()
        graph.add_nodes_from(worker_ids)
        for connection in network_connections:
            graph.add_edge(
                connection.source,
                connection.target,
                connection=connection,
            )

        self._graph = cast(_NetworkGraph, nx.freeze(cast(Any, graph)))

    def workers(self) -> frozenset[WorkerId]:
        return frozenset(self._graph.nodes)

    def successors(self, source: WorkerId) -> frozenset[WorkerId]:
        self._require_worker(source)
        return frozenset(self._graph.successors(source))

    def predecessors(self, target: WorkerId) -> frozenset[WorkerId]:
        self._require_worker(target)
        return frozenset(self._graph.predecessors(target))

    def connections(self) -> Mapping[NetworkConnectionKey, NetworkConnection]:
        connections = cast(
            Iterable[tuple[WorkerId, WorkerId, NetworkConnection]],
            self._graph.edges(data="connection"),
        )
        return {
            (source, target): connection for source, target, connection in connections
        }

    def _require_worker(self, worker_id: WorkerId) -> None:
        if worker_id not in self._graph:
            raise ValueError(f"Unknown worker: {worker_id.service_id}")
