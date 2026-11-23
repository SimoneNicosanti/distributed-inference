from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.adapters.outbound.model_max_cut_size_computer.pulp_max_cut_size_computer import (
    PulpMaxCutSizeComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_invariant_properties_computer import (
    OnnxInvariantPropertiesComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_model_topology_builder import (
    OnnxModelTopologyBuilder,
)
from model_manager.adapters.outbound.model_profiler.onnx_shape_profile_computer import (
    OnnxShapeProfileComputer,
)
from model_manager.application.default_model_profiler import DefaultModelProfiler
from model_manager.application.ports.outbound.contraction_computer import (
    ContractionComputer,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import CNNArchitectureInfo, ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)


@pytest.fixture
def profiled_variant(tmp_path: Path) -> ProfiledModelVariant:
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["features", "mask"], ["sum"], name="sum"),
                helper.make_node(
                    "MatMul", ["sum", "weights"], ["output"], name="linear"
                ),
            ],
            "contract",
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 3])
                for name in ("features", "mask")
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
            [numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), "weights")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    variant = ModelVariant(
        ModelVariantId(ModelId("contract"), "fp32"),
        ModelVariantInfo(
            precision=NumericPrecision.FP32,
            quantization=QuantizationType.NONE,
            accuracies=(),
            format=ModelVariantFormat.ONNX,
            inputs_info={
                name: InputInfo(("batch", 3)) for name in ("features", "mask")
            },
            dynamic_dimension_values={"batch": DynamicShapeValues(frozenset({1, 2}))},
        ),
    )
    contractions = Mock(spec=ContractionComputer)
    contractions.compute_contractions.return_value = [("sum", "linear")]
    profiler = DefaultModelProfiler(
        OnnxModelTopologyBuilder(),
        OnnxInvariantPropertiesComputer(),
        OnnxShapeProfileComputer(PulpMaxCutSizeComputer()),
        contractions,
    )
    profile = profiler._profile_model_sync(
        ArtifactWorkspace(tmp_path, model_path),
        ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo()),
        variant.model_variant_info,
    )
    return ProfiledModelVariant(variant, profile, ArtifactRef("contract-artifact"))


@pytest.mark.asyncio
async def test_redis_profile_contract_from_model_manager_to_worker_and_controller(
    profiled_variant: ProfiledModelVariant,
) -> None:
    from dataclasses import replace

    from fakeredis.aioredis import FakeRedis

    from controller.adapters.outbound.profile.redis_static_profile_store_gateway import (
        RedisStaticProfileStoreGateway,
    )
    from model_manager.adapters.outbound.model_metadata_store.redis_model_metadata_store import (
        RedisModelMetadataStore,
    )
    from model_manager.domain.model import Model
    from redis_access.address_factory.model.address_factory import (
        KeyModelAddressFactory,
    )
    from redis_access.address_factory.model_partition.address_factory import (
        HashMapModelPartitionAddressFactory,
    )
    from redis_access.address_factory.profiled_model_variant.address_factory import (
        HashMapProfiledModelVariantAddressFactory,
    )
    from redis_access.redis_reader.redis.hash_map_redis_reader import (
        HashMapRedisReader,
    )
    from redis_access.redis_reader.redis.key_redis_reader import KeyRedisReader
    from redis_access.redis_reader.redis.set_redis_reader import SetRedisReader
    from worker.adapters.outbound.profiling.reader.redis_model_static_profile_reader import (
        RedisModelStaticProfileReader,
    )

    async with FakeRedis() as redis:
        store = RedisModelMetadataStore(
            redis,
            KeyModelAddressFactory,
            HashMapProfiledModelVariantAddressFactory,
            HashMapModelPartitionAddressFactory,
            KeyRedisReader(),
            HashMapRedisReader(),
            model_existence_checker=KeyRedisReader(),
            variant_existence_checker=HashMapRedisReader(),
        )
        variant_id = profiled_variant.model_variant_id
        assert not await store.check_profiled_model_variant_existence(variant_id)
        await store.add_model(
            Model(
                variant_id.model_id,
                ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo()),
            )
        )
        await store.add_profiled_model_variant(profiled_variant)
        assert await store.check_profiled_model_variant_existence(variant_id)
        restored = await store.get_profiled_model_variant(variant_id)
        assert restored.artifact_ref == profiled_variant.artifact_ref
        assert (
            restored.profile.topology.edges()
            == profiled_variant.profile.topology.edges()
        )
        worker_reader = RedisModelStaticProfileReader(
            redis,
            HashMapProfiledModelVariantAddressFactory,
            HashMapRedisReader(),
            ids_reader=SetRedisReader(),
        )
        assert await worker_reader.get_all_model_static_profile_ids() == [variant_id]
        assert (
            await worker_reader.get_model_static_profile(variant_id)
        ).artifact_ref == profiled_variant.artifact_ref
        controller_reader = RedisStaticProfileStoreGateway(
            redis,
            KeyModelAddressFactory,
            HashMapProfiledModelVariantAddressFactory,
            KeyRedisReader(),
            HashMapRedisReader(),
            ids_reader=SetRedisReader(),
        )
        profiles = await controller_reader.get_all_model_variant_static_profiles()
        assert len(profiles) == 1
        assert profiles[0].model_variant_id == variant_id
        updated = replace(
            profiled_variant, artifact_ref=ArtifactRef("updated-artifact")
        )
        await store.add_profiled_model_variant(updated)
        assert (
            await worker_reader.get_model_static_profile(variant_id)
        ).artifact_ref == updated.artifact_ref
