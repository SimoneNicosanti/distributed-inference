from typing import override

from redis import asyncio as asyncio_redis

from integration.shared.id_json import model_variant_id_to_json
from model_manager.application.ports.outbound.available_model_variant_producer import (
    AvailableModelVariantProducer,
)
from shared.model.model_variant import ModelVariantId


class RedisAvailableModelVariantProducer(AvailableModelVariantProducer):
    def __init__(self, redis: asyncio_redis.Redis) -> None:
        super().__init__()
        self._redis = redis

    @override
    async def publish_available_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> None:

        await self._redis.xadd(
            name="available-model-variants",
            fields={
                "model_variant_id": model_variant_id_to_json(model_variant_id),
            },
            id="*",
        )
