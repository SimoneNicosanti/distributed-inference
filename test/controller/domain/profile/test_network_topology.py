from uuid import UUID

import pytest

from controller.domain.profile.network_topology import (
    NetworkConnection,
    NetworkTopology,
)
from shared.service.service import ServiceId


def _worker(number: int) -> ServiceId:
    return ServiceId(id=UUID(int=number))


def test_exposes_directed_adjacency_without_self_connections() -> None:
    worker_a = _worker(1)
    worker_b = _worker(2)
    worker_c = _worker(3)
    topology = NetworkTopology(
        workers=(worker_a, worker_b, worker_c),
        connections=(
            NetworkConnection(
                source=worker_a,
                target=worker_b,
                throughput_bytes_per_sec=1_000,
                round_trip_time_s=0.01,
            ),
            NetworkConnection(
                source=worker_c,
                target=worker_a,
                throughput_bytes_per_sec=2_000,
                round_trip_time_s=0.02,
            ),
        ),
    )

    assert topology.successors(worker_a) == frozenset({worker_b})
    assert topology.successors(worker_b) == frozenset()
    assert topology.predecessors(worker_a) == frozenset({worker_c})
    assert topology.predecessors(worker_b) == frozenset({worker_a})


def test_adjacency_queries_reject_unknown_workers_without_leaking_networkx() -> None:
    known = _worker(1)
    unknown = _worker(2)
    topology = NetworkTopology(workers=(known,), connections=())

    with pytest.raises(ValueError, match="Unknown worker"):
        topology.successors(unknown)
    with pytest.raises(ValueError, match="Unknown worker"):
        topology.predecessors(unknown)
