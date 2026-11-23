import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from controller.adapters.outbound.profile.redis_execution_profile_store_gateway import (
    RedisExecutionProfileStoreGateway,
)
from controller.adapters.outbound.profile.redis_network_profile_store_gateway import (
    RedisNetworkProfileStoreGateway,
)
from controller.adapters.outbound.profile.redis_resource_profile_store_gateway import (
    RedisResourceProfileStoreGateway,
)
from model_manager.adapters.outbound.model_metadata_store.redis_model_metadata_store import (
    RedisModelMetadataStore,
)
from model_manager.domain.model import Model, ModelInfo
from model_manager.domain.model_partition import ModelPartition
from redis_access.address_factory.model.address_factory import KeyModelAddressFactory
from redis_access.address_factory.model_partition.address_factory import (
    HashMapModelPartitionAddressFactory,
)
from redis_access.address_factory.profiled_model_variant.address_factory import (
    HashMapProfiledModelVariantAddressFactory,
)
from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    HashMapExecutionProfileAddressFactory,
)
from redis_access.address_factory.worker_profiles.network_profile_address_factory import (
    HashMapNetworkProfileAddressFactory,
)
from redis_access.address_factory.worker_profiles.resource_profile_address_factory import (
    HashMapResourceProfileAddressFactory,
)
from redis_access.redis_reader.redis.hash_map_redis_reader import (
    HashMapRedisReader,
)
from redis_access.redis_reader.redis.key_redis_reader import KeyRedisReader
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import CNNArchitectureInfo, ModelId, ModelTask
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from worker.adapters.outbound.model_metadata.redis_model_partition_metadata_reader import (
    RedisModelPartitionMetadataReader,
)
from worker.adapters.outbound.profiling.publisher.redis_network_profile_publisher import (
    RedisNetworkProfilePublisher,
)
from worker.adapters.outbound.profiling.publisher.redis_resource_profile_publisher import (
    RedisResourceProfilePublisher,
)
from worker.adapters.outbound.profiling.publisher.redis_model_execution_profile_publisher import (
    RedisModelExecutionProfilePublisher,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfileWriteError,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
)
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from worker.domain.profiling.network_profile import (
    ConnectionInfo,
    NetworkProfile,
    WorkerConnection,
)
from worker.domain.profiling.resource_profile import (
    CpuProfile,
    RamProfile,
    ResourceProfile,
)


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis]:
    async with FakeRedis() as connection:
        yield connection


def _execution(variant: ModelVariantId, duration: float = 0.1) -> ModelExecutionProfile:
    return ModelExecutionProfile(
        variant,
        (
            ShapeExecutionProfile(
                ModelInputConcreteShapes({"input": (1,)}),
                (
                    BackendExecutionProfile(
                        Backend.CPU, {"layer": LayerExecutionProfile(duration, 10)}
                    ),
                ),
            ),
        ),
    )


def _store(redis: FakeRedis) -> RedisModelExecutionProfilePublisher:
    return RedisModelExecutionProfilePublisher(
        redis,
        HashMapExecutionProfileAddressFactory,
        fields_reader=HashMapRedisReader(),
        existence_checker=HashMapRedisReader(),
    )


@pytest.mark.asyncio
async def test_resource_and_network_round_trip_and_updates_are_fresh(
    redis: FakeRedis,
) -> None:
    first, second, missing = (
        ServiceId(id="first"),
        ServiceId(id="second"),
        ServiceId(id="missing"),
    )
    resource = ResourceProfile(
        (), CpuProfile(1, 1.0, None, None), RamProfile(100, 10, 90, 10)
    )
    publisher = RedisResourceProfilePublisher(
        first, redis, HashMapResourceProfileAddressFactory
    )
    await publisher.publish_resource_profile(resource)
    gateway = RedisResourceProfileStoreGateway(
        redis, HashMapResourceProfileAddressFactory, batch_reader=HashMapRedisReader()
    )
    assert len(await gateway.get_resource_profiles_for_workers([first, missing])) == 1
    await publisher.publish_resource_profile(
        replace(resource, ram_profile=RamProfile(100, 50, 50, 10))
    )
    assert (await gateway.get_resource_profiles_for_workers([first]))[
        0
    ].ram_profile.available_bytes == 50
    stored = await redis.hget("worker-profiles:resource", "first")
    assert b'"available_bytes":50' in stored
    network_publisher = RedisNetworkProfilePublisher(
        first, redis, HashMapNetworkProfileAddressFactory
    )
    await network_publisher.publish_network_profile(
        NetworkProfile((WorkerConnection(second, ConnectionInfo(1000, 0.01)),))
    )
    network = RedisNetworkProfileStoreGateway(
        redis, HashMapNetworkProfileAddressFactory, batch_reader=HashMapRedisReader()
    )
    connections = await network.get_network_connections_by_workers(
        [first, second, missing]
    )
    assert len(connections) == 1
    assert connections[0].throughput_bytes_per_sec == 1000
    await network_publisher.publish_network_profile(
        NetworkProfile((WorkerConnection(second, ConnectionInfo(2000, 0.02)),))
    )
    assert (await network.get_network_connections_by_workers([first, second]))[
        0
    ].throughput_bytes_per_sec == 2000
    assert await network.get_network_connections_by_workers([first]) == []


@pytest.mark.asyncio
async def test_execution_profiles_are_isolated_by_worker_and_variant(
    redis: FakeRedis,
) -> None:
    from unittest.mock import patch

    worker, other = ServiceId(id="worker"), ServiceId(id="other")
    first, second = (
        ModelVariantId(ModelId("one"), "tag"),
        ModelVariantId(ModelId("two"), "tag"),
    )
    missing = ModelVariantId(ModelId("missing"), "tag")
    store = _store(redis)
    batch_reader = HashMapRedisReader()
    gateway = RedisExecutionProfileStoreGateway(
        redis, HashMapExecutionProfileAddressFactory, batch_reader=batch_reader
    )
    assert await store.get_all_profiled_model_ids_by_worker_id(worker) == []
    await store.puplish_model_execution_profile(worker, _execution(first))
    await store.puplish_model_execution_profile(worker, _execution(second))
    await store.puplish_model_execution_profile(other, _execution(first, 0.3))
    assert set(await store.get_all_profiled_model_ids_by_worker_id(worker)) == {
        first,
        second,
    }
    profiles = await gateway.get_execution_profiles_for_workers_and_variants(
        [worker], [first, second]
    )
    assert len(profiles[0].execution_profiles) == 2
    assert await store.check_model_execution_profile_exists(worker, first)
    assert not await store.check_model_execution_profile_exists(worker, missing)
    profile = profiles[0].execution_profiles[first]
    assert (
        profile.shape_execution_profiles[0]
        .backend_profiles[0]
        .layer_profiles["layer"]
        .execution_time_s
        == 0.1
    )
    await store.puplish_model_execution_profile(worker, _execution(first, 0.5))
    assert len(await store.get_all_profiled_model_ids_by_worker_id(worker)) == 2
    with (
        patch.object(batch_reader, "read_many", wraps=batch_reader.read_many) as reads,
        patch.object(redis, "pipeline", wraps=redis.pipeline) as pipelines,
    ):
        profiles = await gateway.get_execution_profiles_for_workers_and_variants(
            [worker, other, worker], [first, missing, first]
        )
        reads.assert_awaited_once()
        assert len(reads.call_args.args[1]) == 4
        pipelines.assert_called_once_with(transaction=False)
    assert all(set(profile.execution_profiles) == {first} for profile in profiles)
    times_by_worker = {
        profile.worker_id: profile.execution_profiles[first]
        .shape_execution_profiles[0]
        .backend_profiles[0]
        .layer_profiles["layer"]
        .execution_time_s
        for profile in profiles
    }
    assert times_by_worker == {worker: 0.5, other: 0.3}
    assert (
        await gateway.get_execution_profiles_for_workers_and_variants([worker], [])
    )[0].execution_profiles == {}


@pytest.mark.asyncio
async def test_concurrent_execution_publications_use_single_writes_without_shared_index(
    redis: FakeRedis,
) -> None:
    from unittest.mock import patch

    workers = [ServiceId(id="first-worker"), ServiceId(id="second-worker")]
    variants = [ModelVariantId(ModelId("model"), str(i)) for i in range(12)]
    with (
        patch.object(
            redis,
            "pipeline",
            side_effect=AssertionError(
                "Publication must not use WATCH or transactions"
            ),
        ),
        patch.object(
            redis,
            "hget",
            side_effect=AssertionError("Publication must not read-modify-write"),
        ),
        patch.object(redis, "hset", wraps=redis.hset) as writes,
    ):
        await asyncio.gather(
            *(
                _store(redis).puplish_model_execution_profile(
                    worker, _execution(variant)
                )
                for worker in workers
                for variant in variants
            )
        )
        assert writes.call_count == len(workers) * len(variants)
    for worker in workers:
        assert set(
            await _store(redis).get_all_profiled_model_ids_by_worker_id(worker)
        ) == set(variants)
    gateway = RedisExecutionProfileStoreGateway(
        redis, HashMapExecutionProfileAddressFactory, batch_reader=HashMapRedisReader()
    )
    profiles = await gateway.get_execution_profiles_for_workers_and_variants(
        workers, variants
    )
    assert all(len(profile.execution_profiles) == len(variants) for profile in profiles)
    assert set(await redis.keys("*")) == {
        b"worker-profiles:execution:first-worker",
        b"worker-profiles:execution:second-worker",
    }


@pytest.mark.asyncio
async def test_profile_ids_follow_hash_fields_without_a_separate_index(
    redis: FakeRedis,
) -> None:
    worker = ServiceId(id="worker")
    variant = ModelVariantId(ModelId("model"), "tag")
    store = _store(redis)
    await store.puplish_model_execution_profile(worker, _execution(variant))
    address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        worker, variant
    )
    await redis.hdel(address.hash_map_key, address.field_key)
    assert await store.get_all_profiled_model_ids_by_worker_id(worker) == []
    assert not await store.check_model_execution_profile_exists(worker, variant)


@pytest.mark.asyncio
async def test_execution_publication_propagates_redis_write_failure(
    redis: FakeRedis,
) -> None:
    worker = ServiceId(id="worker")
    variant = ModelVariantId(ModelId("model"), "tag")
    container = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
        worker
    )
    await redis.set(container.key, "wrong-type")
    with pytest.raises(ModelExecutionProfileWriteError):
        await _store(redis).puplish_model_execution_profile(worker, _execution(variant))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["fields", "exists"])
async def test_execution_store_preserves_read_error_translation(
    redis: FakeRedis, operation: str
) -> None:
    from redis.exceptions import ResponseError
    from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_reader import (
        ModelExecutionProfileReadError,
    )

    from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError

    worker = ServiceId(id="worker")
    variant = ModelVariantId(ModelId("model"), "tag")
    container = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
        worker
    )
    await redis.set(container.key, "wrong-type")

    with pytest.raises(ModelExecutionProfileReadError) as caught:
        if operation == "fields":
            await _store(redis).get_all_profiled_model_ids_by_worker_id(worker)
        else:
            await _store(redis).check_model_execution_profile_exists(worker, variant)

    assert isinstance(caught.value.__cause__, RedisReadError)
    assert isinstance(caught.value.__cause__.__cause__, ResponseError)


@pytest.mark.asyncio
async def test_model_and_partition_metadata_share_addresses_and_remain_fresh(
    redis: FakeRedis,
) -> None:
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
    model = Model(
        ModelId("model"), ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo())
    )
    assert not await store.check_model_existence(model.model_id)
    await store.add_model(model)
    assert await store.check_model_existence(model.model_id)
    assert await store.get_model(model.model_id) == model
    partition = ModelPartition(
        ModelPartitionId(ModelVariantId(model.model_id, "tag"), ("layer",)),
        ArtifactRef("first"),
    )
    reader = RedisModelPartitionMetadataReader(
        redis, HashMapRedisReader(), HashMapModelPartitionAddressFactory()
    )
    await store.add_model_partition(partition)
    assert await reader.get_artifact_ref_by_partition_id(
        partition.model_partition_id
    ) == ArtifactRef("first")
    await store.add_model_partition(
        replace(partition, artifact_ref=ArtifactRef("second"))
    )
    assert await reader.get_artifact_ref_by_partition_id(
        partition.model_partition_id
    ) == ArtifactRef("second")


@pytest.mark.asyncio
async def test_execution_gateway_rejects_profile_under_another_variant_field(
    redis: FakeRedis,
) -> None:
    from worker.adapters.outbound.profiling.publisher.mappers.model_execution_profile_mapper import (
        ModelExecutionProfileMapper,
    )

    worker = ServiceId(id="worker")
    expected = ModelVariantId(ModelId("expected"), "tag")
    other = ModelVariantId(ModelId("other"), "tag")
    address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        worker, expected
    )
    payload = ModelExecutionProfileMapper.model_execution_profile_to_dto(
        worker, _execution(other)
    ).model_dump_json()
    await redis.hset(address.hash_map_key, address.field_key, payload)
    gateway = RedisExecutionProfileStoreGateway(
        redis, HashMapExecutionProfileAddressFactory, batch_reader=HashMapRedisReader()
    )
    with pytest.raises(ValueError, match="variant does not match"):
        await gateway.get_execution_profiles_for_workers_and_variants(
            [worker], [expected, other]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["key", "hash"])
async def test_readers_distinguish_missing_values_from_redis_failures(
    redis: FakeRedis, kind: str
) -> None:
    from unittest.mock import patch

    from redis.exceptions import ConnectionError as RedisConnectionError

    from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
    from redis_access.redis_reader.abc.single_value_redis_reader import (
        RedisReadError,
        RedisValueNotFoundError,
    )

    reader, address, method = (
        (KeyRedisReader(), KeyRedisAddress("missing"), "get")
        if kind == "key"
        else (
            HashMapRedisReader(),
            HashMapRedisAddress("missing", "field"),
            "hget",
        )
    )
    with pytest.raises(RedisValueNotFoundError):
        await reader.read_single(redis, address)
    with patch.object(redis, method, side_effect=RedisConnectionError("offline")):
        with pytest.raises(RedisReadError) as error:
            await reader.read_single(redis, address)
        assert not isinstance(error.value, RedisValueNotFoundError)
        assert isinstance(error.value.__cause__, RedisConnectionError)


@pytest.mark.asyncio
async def test_model_store_does_not_report_backend_failure_as_missing_model(
    redis: FakeRedis,
) -> None:
    from unittest.mock import patch

    from redis.exceptions import ConnectionError as RedisConnectionError

    from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError

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
    with pytest.raises(ValueError, match="not found"):
        await store.get_model(ModelId("missing"))
    with (
        patch.object(redis, "get", side_effect=RedisConnectionError("offline")),
        pytest.raises(RedisReadError),
    ):
        await store.get_model(ModelId("missing"))
