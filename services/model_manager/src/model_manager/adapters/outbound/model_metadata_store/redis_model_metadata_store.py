from typing import override

from redis import asyncio as redis_asyncio

from integration.redis.model.keys_factory import build_model_key
from integration.redis.model_partition.keys_factory import (
    build_model_partition_dict_name,
    build_model_partition_key,
)
from integration.redis.profiled_model_variant.keys_factory import (
    build_profiled_model_variant_dict_name,
    build_profiled_model_variant_ids_key,
    build_profiled_model_variant_key,
)
from integration.redis.profiled_model_variant.profiled_model_variant_dto import (
    EdgeDescriptionDto,
    InfoDto,
    LayerDescriptionDto,
    LayerPropertyDto,
    LayerShapePropertyDto,
    ProfiledModelVariantDto,
    ProfileDto,
    ShapePointDto,
    ShapeProfileDto,
    TensorPropertyDto,
    TensorShapePropertyDto,
    TopologyDto,
)
from model_manager.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from model_manager.domain.layer_property import LayerProperty
from model_manager.domain.model import Model
from model_manager.domain.model_partition import ModelPartition
from model_manager.domain.model_variant import ModelVariant, ModelVariantInfo
from model_manager.domain.model_variant_profile import ModelVariantProfile
from model_manager.domain.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from model_manager.domain.shape_profile import ShapePoint, ShapeProfile
from model_manager.domain.tensor_property import TensorProperty
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class RedisModelMetadataStore(ModelMetadataStore):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def add_model(self, model: Model) -> None:
        model_key = await build_model_key(model.model_id)

        await self._redis.set(model_key, model.model_dump_json())

    @override
    async def get_model(self, model_id: ModelId) -> Model:
        model_key = await build_model_key(model_id)

        value = await self._redis.get(model_key)
        if value is None:
            raise ValueError(f"Model {model_id} not found")

        return Model.model_validate_json(value)

    @override
    async def check_model_existence(self, model_id: ModelId) -> bool:
        model_key = await build_model_key(model_id)

        num_values = await self._redis.exists(model_key)
        return num_values > 0

    @override
    async def add_profiled_model_variant(
        self,
        profiled_model_variant: ProfiledModelVariant,
    ) -> None:
        set_name = await build_profiled_model_variant_dict_name(
            profiled_model_variant.model_variant_id
        )
        profiled_model_variant_key = await build_profiled_model_variant_key(
            profiled_model_variant.model_variant_id
        )
        dto = self._to_profiled_model_variant_dto(profiled_model_variant)
        dto_json = dto.model_dump_json()
        model_variant_id_json = (
            profiled_model_variant.model_variant_id.model_dump_json()
        )

        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                set_name,
                profiled_model_variant_key,
                dto_json,
            )
            pipeline.sadd(
                build_profiled_model_variant_ids_key(),
                model_variant_id_json,
            )
            await pipeline.execute()

    @override
    async def get_profiled_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> ProfiledModelVariant:
        set_name = await build_profiled_model_variant_dict_name(model_variant_id)
        profiled_model_variant_key = await build_profiled_model_variant_key(
            model_variant_id
        )

        value = await self._redis.hget(set_name, profiled_model_variant_key)

        if value is None:
            raise ValueError(f"Profiled model variant {model_variant_id} not found")

        dto = ProfiledModelVariantDto.model_validate_json(value)
        return self._from_profiled_model_variant_dto(dto)

    @override
    async def check_profiled_model_variant_existence(
        self,
        model_variant_id: ModelVariantId,
    ) -> bool:
        set_name = await build_profiled_model_variant_dict_name(model_variant_id)
        profiled_model_variant_key = await build_profiled_model_variant_key(
            model_variant_id
        )

        exists = await self._redis.hexists(set_name, profiled_model_variant_key)
        return exists

    @override
    async def add_model_partition(
        self,
        model_partition: ModelPartition,
    ) -> None:
        set_name = await build_model_partition_dict_name(
            model_partition.model_partition_id
        )
        partition_key = await build_model_partition_key(
            model_partition.model_partition_id
        )

        await self._redis.hset(
            set_name,
            partition_key,
            model_partition.model_dump_json(),
        )

    @staticmethod
    def _to_profiled_model_variant_dto(
        profiled_model_variant: ProfiledModelVariant,
    ) -> ProfiledModelVariantDto:
        profile = profiled_model_variant.profile

        topology = TopologyDto(
            layers={
                layer_key: LayerDescriptionDto.model_validate(description.model_dump())
                for layer_key, description in profile.topology.layers().items()
            },
            edges=[
                EdgeDescriptionDto.model_validate(description.model_dump())
                for description in profile.topology.edges().values()
            ],
        )

        shape_profiles = [
            ShapeProfileDto(
                shape_point=ShapePointDto.model_validate(shape_point.model_dump()),
                layer_properties={
                    layer_key: LayerShapePropertyDto.model_validate(
                        property_.model_dump()
                    )
                    for layer_key, property_ in shape_profile.layer_properties.items()
                },
                tensor_properties={
                    tensor_key: TensorShapePropertyDto.model_validate(
                        property_.model_dump()
                    )
                    for tensor_key, property_ in shape_profile.tensor_properties.items()
                },
            )
            for shape_point, shape_profile in profile.shape_profiles.items()
        ]

        return ProfiledModelVariantDto(
            model_variant_id=profiled_model_variant.model_variant_id,
            info=InfoDto.model_validate(
                profiled_model_variant.model_variant.model_variant_info.model_dump()
            ),
            profile=ProfileDto(
                topology=topology,
                invariant_layer_properties={
                    layer_key: LayerPropertyDto.model_validate(property_.model_dump())
                    for layer_key, property_ in (
                        profile.invariant_layer_properties.items()
                    )
                },
                invariant_tensor_properties={
                    tensor_key: TensorPropertyDto.model_validate(property_.model_dump())
                    for tensor_key, property_ in (
                        profile.invariant_tensor_properties.items()
                    )
                },
                shape_profiles=shape_profiles,
                optimization_contractions=profile.optimization_contractions,
            ),
            artifact_ref=profiled_model_variant.artifact_ref,
        )

    @staticmethod
    def _from_profiled_model_variant_dto(
        dto: ProfiledModelVariantDto,
    ) -> ProfiledModelVariant:
        topology = ModelVariantTopology()
        for layer_key, layer_description in dto.profile.topology.layers.items():
            topology.add_layer(
                layer_key,
                LayerDescription.model_validate(layer_description.model_dump()),
            )
        for edge_description in dto.profile.topology.edges:
            topology.add_edge(
                (edge_description.source, edge_description.target),
                EdgeDescription.model_validate(edge_description.model_dump()),
            )

        shape_profiles: dict[ShapePoint, ShapeProfile] = {}
        for shape_profile_dto in dto.profile.shape_profiles:
            shape_point = ShapePoint.model_validate(
                shape_profile_dto.shape_point.model_dump()
            )
            if shape_point in shape_profiles:
                raise ValueError(f"Duplicate shape point in Redis DTO: {shape_point}")

            shape_profiles[shape_point] = ShapeProfile.model_validate(
                {
                    "layer_properties": {
                        layer_key: property_.model_dump()
                        for layer_key, property_ in (
                            shape_profile_dto.layer_properties.items()
                        )
                    },
                    "tensor_properties": {
                        tensor_key: property_.model_dump()
                        for tensor_key, property_ in (
                            shape_profile_dto.tensor_properties.items()
                        )
                    },
                }
            )

        return ProfiledModelVariant(
            model_variant=ModelVariant(
                model_variant_id=dto.model_variant_id,
                model_variant_info=ModelVariantInfo.model_validate(
                    dto.info.model_dump()
                ),
            ),
            profile=ModelVariantProfile(
                topology=topology,
                invariant_layer_properties={
                    layer_key: LayerProperty.model_validate(property_.model_dump())
                    for layer_key, property_ in (
                        dto.profile.invariant_layer_properties.items()
                    )
                },
                invariant_tensor_properties={
                    tensor_key: TensorProperty.model_validate(property_.model_dump())
                    for tensor_key, property_ in (
                        dto.profile.invariant_tensor_properties.items()
                    )
                },
                shape_profiles=shape_profiles,
                optimization_contractions=dto.profile.optimization_contractions,
            ),
            artifact_ref=dto.artifact_ref,
        )
