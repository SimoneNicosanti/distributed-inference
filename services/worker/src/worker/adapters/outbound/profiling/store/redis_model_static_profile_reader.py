import logging
from typing import override

from pydantic import ValidationError
from redis import RedisError
from redis import asyncio as redis_asyncio

from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
)
from integration.shared.id_json import model_variant_id_from_json
from redis_access.address_factory.profiled_model_variant.address_factory import (
    ProfiledModelVariantAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.redis_reader import RedisReader, RedisReadError
from shared.model.model_variant import ModelVariantId
from worker.adapters.outbound.profiling.store.mappers.model_static_profile_mapper import (
    ModelStaticProfileMapper,
)
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
    ModelStaticProfileReadError,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)

logger = logging.getLogger(__name__)


class RedisModelStaticProfileReader(ModelStaticProfileReader):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        profiled_model_variant_address_factory_cls: type[
            ProfiledModelVariantAddressFactory[HashMapRedisAddress]
        ],
        redis_reader: RedisReader[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._redis_reader = redis_reader
        self._profiled_model_variant_address_factory_cls = (
            profiled_model_variant_address_factory_cls
        )

    @override
    async def get_all_model_static_profile_ids(self) -> list[ModelVariantId]:
        ids_key = self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_ids_address().key

        try:
            ids: list[ModelVariantId] = []
            async for value in self._redis.sscan_iter(ids_key):
                self._append_id_to_list_from_value(ids, value)
            return ids
        except RedisError as e:
            raise ModelStaticProfileReadError(
                "Unable to read model static profile IDs"
            ) from e

    def _append_id_to_list_from_value(
        self, ids: list[ModelVariantId], value: str | bytes
    ) -> None:
        try:
            id = model_variant_id_from_json(value)
            ids.append(id)
        except ValidationError:
            logger.exception("Skipping wrong formatted model variant id")

    @override
    async def get_model_static_profile(
        self,
        model_variant_id: ModelVariantId,
    ) -> ModelStaticProfile:
        try:
            addr = await self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_address(
                model_variant_id
            )
            value = await self._redis_reader.read(self._redis, addr)

            dto = ProfiledModelVariantDto.model_validate_json(value)
            if dto.model_variant_id != model_variant_id:
                raise ValueError("Stored model variant ID mismatch")
            return ModelStaticProfileMapper.model_static_profile_from_dto(dto)
        except (ValueError, RedisError, RedisReadError) as e:
            raise ModelStaticProfileReadError(
                f"Unable to read model static profile {model_variant_id}"
            ) from e
