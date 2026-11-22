import hashlib

import pytest

from redis_access.address_factory.model_partition.address_factory import (
    HashMapModelPartitionAddressFactory,
)
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "variant_tag", "layers", "serialized_id"),
    [
        (
            "model",
            "variant",
            ("layer-b", "layer-a"),
            '{"model_variant_id":{"model_id":{"model_name":"model"},'
            '"variant_tag":"variant"},"layers":["layer-a","layer-b"]}',
        ),
        (
            "模型",
            "é🚀",
            ("東京",),
            '{"model_variant_id":{"model_id":{"model_name":"模型"},'
            '"variant_tag":"é🚀"},"layers":["東京"]}',
        ),
        (
            'model"name',
            "tag\nvalue",
            ("layer\tname",),
            '{"model_variant_id":{"model_id":{"model_name":"model\\"name"},'
            '"variant_tag":"tag\\nvalue"},"layers":["layer\\tname"]}',
        ),
    ],
)
async def test_partition_address_preserves_serialized_identity(
    model_name: str,
    variant_tag: str,
    layers: tuple[str, ...],
    serialized_id: str,
) -> None:
    variant_id = ModelVariantId(ModelId(model_name), variant_tag)
    partition_id = ModelPartitionId(variant_id, layers)

    address = await HashMapModelPartitionAddressFactory.build_model_partition_address(
        partition_id
    )

    assert address.hash_map_key == f"model-partitions:{variant_id.full_variant_tag}"
    assert address.field_key == hashlib.md5(serialized_id.encode("utf-8")).hexdigest()
