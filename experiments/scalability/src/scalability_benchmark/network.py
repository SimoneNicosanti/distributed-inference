"""Full directed topology; reciprocal links share sampled physical parameters."""

from controller.domain.profile.network_topology import (
    NetworkConnection,
    NetworkTopology,
)
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.hardware import SyntheticWorker
from scalability_benchmark.randomness import stream


def network_pool(
    config: ExperimentConfig, seed: int, workers: tuple[SyntheticWorker, ...]
) -> tuple[NetworkConnection, ...]:
    connections: list[NetworkConnection] = []
    tier_order = {"device": 0, "edge": 1, "cloud": 2}
    for index, left in enumerate(workers):
        for right in workers[index + 1 :]:
            tiers = sorted((left.tier, right.tier), key=tier_order.__getitem__)
            settings = config.network["-".join(tiers)]
            keys = sorted((str(left.id.id), str(right.id.id)))
            rng = stream(seed, "network", *keys)
            latency = rng.uniform(*settings.one_way_ms) / 1000
            throughput = rng.uniform(*settings.throughput_mbps) * 1_000_000 / 8
            connections.extend(
                (
                    NetworkConnection(left.id, right.id, throughput, 2 * latency),
                    NetworkConnection(right.id, left.id, throughput, 2 * latency),
                )
            )
    return tuple(connections)


def select_network(
    workers: tuple[SyntheticWorker, ...], pool: tuple[NetworkConnection, ...]
) -> NetworkTopology:
    identities = {worker.id for worker in workers}
    return NetworkTopology(
        workers=identities,
        connections=(
            connection
            for connection in pool
            if connection.source in identities and connection.target in identities
        ),
    )
