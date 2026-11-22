import json

import pytest

from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    HashMapExecutionProfileAddressFactory,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId


@pytest.mark.asyncio
async def test_profile_address_preserves_complete_identity() -> None:
    worker = ServiceId(id='worker:"東京"')
    variant = ModelVariantId(ModelId("model-模型:名字"), 'variant:"é"')

    address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        worker, variant
    )
    repeated = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        worker, variant
    )

    assert address.hash_map_key == f"worker-profiles:execution:{worker.id}"
    assert json.loads(address.field_key) == {
        "model_id": {"model_name": variant.model_id.model_name},
        "variant_tag": variant.variant_tag,
    }
    assert address == repeated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("worker", "a-b", "c"), ("worker", "a", "b-c")),
        (("a:b", "c", "d"), ("a", "b:c", "d")),
        (("first-worker", "model", "variant"), ("second-worker", "model", "variant")),
    ],
)
async def test_distinct_profile_identities_have_distinct_addresses(
    first: tuple[str, str, str], second: tuple[str, str, str]
) -> None:
    first_address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        ServiceId(id=first[0]), ModelVariantId(ModelId(first[1]), first[2])
    )
    second_address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        ServiceId(id=second[0]), ModelVariantId(ModelId(second[1]), second[2])
    )

    assert first_address != second_address


@pytest.mark.asyncio
async def test_profiles_share_worker_container_and_isolate_other_workers() -> None:
    worker = ServiceId(id="worker")
    profile_address = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        worker, ModelVariantId(ModelId("profiled"), "ids")
    )
    container = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
        worker
    )
    other = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
        ServiceId(id="other-worker")
    )
    assert container.key == "worker-profiles:execution:worker"
    assert container.key == profile_address.hash_map_key
    assert other.key != container.key
