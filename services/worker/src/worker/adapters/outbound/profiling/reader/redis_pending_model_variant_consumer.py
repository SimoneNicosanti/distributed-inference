from collections.abc import AsyncGenerator
from typing import override

from redis import asyncio as asyncio_redis
from redis.exceptions import ResponseError

from integration.shared.id_json import model_variant_id_from_json
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.model_execution.pending_model_variant_consumer import (
    PendingModelVariantConsumer,
)


class RedisPendingModelVariantConsumer(PendingModelVariantConsumer):
    def __init__(self, worker_id: WorkerId, redis: asyncio_redis.Redis) -> None:
        super().__init__()
        self._worker_id = worker_id
        self._redis = redis

        self._stream = "available-model-variants"
        self._group = f"profiling:{worker_id.id}"
        self._consumer = "profiler"
        self._message_ids_map: dict[ModelVariantId, set[str | bytes]] = {}

    @override
    async def pending_model_variant_ids(self) -> AsyncGenerator[ModelVariantId]:
        try:
            await self._redis.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
        except ResponseError as error:
            if not str(error).startswith("BUSYGROUP"):
                raise

        pending = await self._redis.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams={self._stream: "0"},
        )
        available = await self._redis.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams={self._stream: ">"},
        )

        # Capture a finite round before yielding, so failures wait for the next round.
        for _, messages in [*pending, *available]:
            for message_id, fields in messages:  # type: ignore
                model_variant_id = model_variant_id_from_json(
                    fields[b"model_variant_id"]
                )
                self._message_ids_map.setdefault(model_variant_id, set()).add(
                    message_id
                )
                yield model_variant_id

    @override
    async def acknowledge(self, model_variant_id: ModelVariantId) -> None:
        message_ids = self._message_ids_map.get(model_variant_id)
        if not message_ids:
            raise RuntimeError("Model variant has no pending delivery")

        await self._redis.xack(
            self._stream,
            self._group,
            *message_ids,
        )
        del self._message_ids_map[model_variant_id]
