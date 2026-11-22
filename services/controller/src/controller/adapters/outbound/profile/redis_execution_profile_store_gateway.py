from collections.abc import Iterable
from typing import override

from redis import asyncio as redis_asyncio

from controller.adapters.outbound.profile.mappers.model_variant_execution_profile_mapper import (
    ModelVariantExecutionProfileMapper,
)
from controller.application.ports.outbound.profile.execution_profile_store_gateway import (
    ExecutionProfileStoreGateway,
)
from controller.domain.profile.model_variant_execution_profile import (
    ModelVariantExecutionProfile,
    WorkerExecutionProfiles,
)
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.shared.id_json import model_variant_id_to_json
from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    ExecutionProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class RedisExecutionProfileStoreGateway(ExecutionProfileStoreGateway):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        execution_profile_address_factory_cls: type[
            ExecutionProfileAddressFactory[HashMapRedisAddress]
        ],
    ) -> None:
        self._redis = redis
        self._execution_profile_address_factory_cls = (
            execution_profile_address_factory_cls
        )

    @override
    async def get_execution_profiles_for_workers_and_variants(
        self,
        worker_ids: Iterable[WorkerId],
        model_variant_ids: Iterable[ModelVariantId],
    ) -> list[WorkerExecutionProfiles]:
        ordered_worker_ids = sorted(
            set(worker_ids),
            key=lambda worker_id: str(worker_id.id),
        )
        ordered_model_variant_ids = sorted(
            set(model_variant_ids),
            key=model_variant_id_to_json,
        )

        if not ordered_worker_ids:
            return []
        if not ordered_model_variant_ids:
            return [
                WorkerExecutionProfiles(worker_id=worker_id, execution_profiles={})
                for worker_id in ordered_worker_ids
            ]

        async with self._redis.pipeline(transaction=False) as pipeline:
            for worker_id in ordered_worker_ids:
                addresses = [
                    await self._execution_profile_address_factory_cls.build_worker_model_execution_profile_address(
                        worker_id, variant_id
                    )
                    for variant_id in ordered_model_variant_ids
                ]
                if len({addr.hash_map_key for addr in addresses}) != 1:
                    raise ValueError(
                        "Execution profiles must share one hash for batch reads"
                    )
                pipeline.hmget(
                    addresses[0].hash_map_key, [addr.field_key for addr in addresses]
                )
            stored_profiles_by_worker = await pipeline.execute()

        return [
            self._build_worker_profiles(
                worker_id,
                stored_profiles,
                ordered_model_variant_ids,
            )
            for worker_id, stored_profiles in zip(
                ordered_worker_ids, stored_profiles_by_worker, strict=True
            )
        ]

    @classmethod
    def _build_worker_profiles(
        cls,
        worker_id: WorkerId,
        stored_profiles: Iterable[str | bytes | bytearray | None],
        ordered_model_variant_ids: list[ModelVariantId],
    ) -> WorkerExecutionProfiles:
        profiles_by_variant: dict[
            ModelVariantId,
            ModelVariantExecutionProfile,
        ] = {}

        for expected_variant_id, value in zip(
            ordered_model_variant_ids, stored_profiles, strict=True
        ):
            if value is None:
                continue
            dto = ModelExecutionProfileDto.model_validate_json(value)
            if dto.worker_id != worker_id:
                raise ValueError(
                    "Stored execution profile worker does not match the requested worker"
                )
            if dto.model_variant_id != expected_variant_id:
                raise ValueError(
                    "Stored execution profile variant does not match its Redis field"
                )
            if dto.model_variant_id in profiles_by_variant:
                raise ValueError(
                    "Duplicate execution profiles for a worker and model variant"
                )

            profiles_by_variant[dto.model_variant_id] = (
                ModelVariantExecutionProfileMapper.model_variant_execution_profile_from_dto(
                    dto
                )
            )

        return WorkerExecutionProfiles(
            worker_id=worker_id,
            execution_profiles={
                model_variant_id: profiles_by_variant[model_variant_id]
                for model_variant_id in ordered_model_variant_ids
                if model_variant_id in profiles_by_variant
            },
        )
