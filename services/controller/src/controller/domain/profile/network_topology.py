from collections.abc import Iterable, Mapping
from typing import Any, TypedDict, cast

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from shared.service.service import WorkerId


class NetworkConnection(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: WorkerId
    target: WorkerId
    throughput_bytes_per_sec: float = Field(gt=0, allow_inf_nan=False)
    round_trip_time_s: float = Field(ge=0, allow_inf_nan=False)


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

    def connections(self) -> Mapping[NetworkConnectionKey, NetworkConnection]:
        connections = cast(
            Iterable[tuple[WorkerId, WorkerId, NetworkConnection]],
            self._graph.edges(data="connection"),
        )
        return {
            (source, target): connection for source, target, connection in connections
        }
