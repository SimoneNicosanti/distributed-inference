import hashlib

import pytest

from integration.redis.worker_profiles.execution_profile_keys import (
    HashExecutionProfileKeyFactory,
    StringExecutionProfileKeyFactory,
)
from integration.redis.worker_profiles.network_profile_keys import (
    HashNetworkProfileKeyFactory,
    StringNetworkProfileKeyFactory,
)
from integration.redis.worker_profiles.resource_profile_keys import (
    HashResourceProfileKeyFactory,
    StringResourceProfileKeyFactory,
)
from integration.shared.id_json import model_variant_id_to_json
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId

WORKER_ID = ServiceId(id="worker-1")
MODEL_VARIANT_ID = ModelVariantId(ModelId("model-a"), "variant-1")


@pytest.mark.asyncio
async def test_string_worker_profile_keys_contain_worker_id() -> None:
    assert (
        await StringExecutionProfileKeyFactory.build_execution_profile_key_per_worker(
            WORKER_ID
        )
        == "worker:worker-1:model-execution-profiles"
    )
    assert (
        await StringExecutionProfileKeyFactory.build_execution_profile_ids_key_per_worker(
            WORKER_ID
        )
        == "worker:worker-1:model-execution-profile-ids"
    )
    assert (
        await StringNetworkProfileKeyFactory.build_network_profile_key_per_worker(
            WORKER_ID
        )
        == "worker:worker-1:network-profile"
    )
    assert (
        await StringResourceProfileKeyFactory.build_resource_profile_key_per_worker(
            WORKER_ID
        )
        == "worker:worker-1:resource-profile"
    )


@pytest.mark.asyncio
async def test_hash_worker_profile_keys_use_same_worker_hash() -> None:
    worker_hash = hashlib.md5(b"worker-1").hexdigest()

    assert (
        await HashExecutionProfileKeyFactory.build_execution_profile_key_per_worker(
            WORKER_ID
        )
        == f"worker:{worker_hash}:model-execution-profiles"
    )
    assert (
        await HashExecutionProfileKeyFactory.build_execution_profile_ids_key_per_worker(
            WORKER_ID
        )
        == f"worker:{worker_hash}:model-execution-profile-ids"
    )
    assert (
        await HashNetworkProfileKeyFactory.build_network_profile_key_per_worker(
            WORKER_ID
        )
        == f"worker:{worker_hash}:network-profile"
    )
    assert (
        await HashResourceProfileKeyFactory.build_resource_profile_key_per_worker(
            WORKER_ID
        )
        == f"worker:{worker_hash}:resource-profile"
    )


@pytest.mark.asyncio
async def test_hash_execution_profile_field_preserves_existing_format() -> None:
    expected = hashlib.md5(
        model_variant_id_to_json(MODEL_VARIANT_ID).encode("utf-8")
    ).hexdigest()

    assert (
        await HashExecutionProfileKeyFactory.build_model_execution_profile_key(
            MODEL_VARIANT_ID
        )
        == expected
    )


@pytest.mark.asyncio
async def test_string_execution_profile_field_uses_canonical_id_json() -> None:
    key = await StringExecutionProfileKeyFactory.build_model_execution_profile_key(
        MODEL_VARIANT_ID
    )

    assert key == model_variant_id_to_json(MODEL_VARIANT_ID)


@pytest.mark.asyncio
async def test_string_execution_profile_field_includes_model_name() -> None:
    other_model_variant_id = ModelVariantId(ModelId("model-b"), "variant-1")

    first_key = (
        await StringExecutionProfileKeyFactory.build_model_execution_profile_key(
            MODEL_VARIANT_ID
        )
    )
    second_key = (
        await StringExecutionProfileKeyFactory.build_model_execution_profile_key(
            other_model_variant_id
        )
    )

    assert first_key != second_key
