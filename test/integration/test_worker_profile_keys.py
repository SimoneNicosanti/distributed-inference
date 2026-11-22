import json

import pytest

from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    HashMapExecutionProfileAddressFactory,
)
from redis_access.address_factory.worker_profiles.network_profile_address_factory import (
    HashMapNetworkProfileAddressFactory,
)
from redis_access.address_factory.worker_profiles.resource_profile_address_factory import (
    HashMapResourceProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId

WORKER_ID = ServiceId(id="worker-1")
MODEL_VARIANT_ID = ModelVariantId(ModelId("model-a"), "variant-1")


def test_resource_and_network_profiles_use_worker_fields() -> None:
    assert HashMapResourceProfileAddressFactory.build_worker_resource_profile_address(
        WORKER_ID
    ) == HashMapRedisAddress("worker-profiles:resource", "worker-1")
    assert HashMapNetworkProfileAddressFactory.build_worker_network_profile_address(
        WORKER_ID
    ) == HashMapRedisAddress("worker-profiles:network", "worker-1")


@pytest.mark.asyncio
async def test_execution_profile_address_contains_complete_identity() -> None:
    addr = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        WORKER_ID, MODEL_VARIANT_ID
    )
    assert addr.hash_map_key == "worker-profiles:execution:worker-1"
    assert json.loads(addr.field_key) == {
        "model_id": {"model_name": "model-a"},
        "variant_tag": "variant-1",
    }
    container = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
        WORKER_ID
    )
    assert container == KeyRedisAddress("worker-profiles:execution:worker-1")


@pytest.mark.asyncio
async def test_execution_profile_addresses_distinguish_models_and_workers() -> None:
    first = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        WORKER_ID, MODEL_VARIANT_ID
    )
    other_model = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        WORKER_ID, ModelVariantId(ModelId("model-b"), "variant-1")
    )
    other_worker = await HashMapExecutionProfileAddressFactory.build_worker_model_execution_profile_address(
        ServiceId(id="worker-2"), MODEL_VARIANT_ID
    )
    assert len({first, other_model, other_worker}) == 3
