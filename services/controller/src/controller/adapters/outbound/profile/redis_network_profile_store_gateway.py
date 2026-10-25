from collections.abc import Iterable
from typing import override

from redis import asyncio as redis_asyncio

from controller.application.ports.outbound.profile.network_profile_store_gateway import (
    NetworkProfileStoreGateway,
)
from controller.domain.profile.network_topology import (
    NetworkConnection,
)
from integration.dto.worker_profiles.network_profile_dto import NetworkProfileDto
from integration.redis.worker_profiles.network_profile_keys import (
    build_network_profile_key_per_worker,
)
from shared.service.service import WorkerId


class RedisNetworkProfileStoreGateway(NetworkProfileStoreGateway):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_network_connections_by_workers(
        self, worker_ids: Iterable[WorkerId]
    ) -> list[NetworkConnection]:
        requested_worker_ids = frozenset(worker_ids)
        if not requested_worker_ids:
            return []

        ordered_worker_ids = sorted(
            requested_worker_ids,
            key=lambda worker_id: str(worker_id.id),
        )
        redis_keys = [
            build_network_profile_key_per_worker(worker_id)
            for worker_id in ordered_worker_ids
        ]
        values = await self._redis.mget(redis_keys)

        connections: list[NetworkConnection] = []
        for worker_id, value in zip(ordered_worker_ids, values, strict=True):
            if value is None:
                continue

            profile = NetworkProfileDto.model_validate_json(value)
            if profile.source_worker_id != worker_id:
                raise ValueError(
                    "Stored network profile source worker does not match "
                    "the requested worker"
                )

            for connection in profile.connections:
                if connection.target_worker_id not in requested_worker_ids:
                    continue
                connections.append(
                    NetworkConnection(
                        source=profile.source_worker_id,
                        target=connection.target_worker_id,
                        throughput_bytes_per_sec=connection.throughput_bytes_per_sec,
                        round_trip_time_s=connection.round_trip_time_s,
                    )
                )
        return connections
