import asyncio
import hashlib

from shared.model.model_partition import ModelPartitionId


async def build_model_partition_dict_name(model_partition_id: ModelPartitionId) -> str:
    model_variant_id_md5 = await asyncio.to_thread(
        hashlib.md5,
        model_partition_id.model_variant_id.model_dump_json().encode("utf-8"),
    )

    return f"model-partitions:{model_variant_id_md5.hexdigest()}"


async def build_model_partition_key(model_partition_id: ModelPartitionId) -> str:
    model_partition_id_md5 = await asyncio.to_thread(
        hashlib.md5,
        model_partition_id.model_dump_json().encode("utf-8"),
    )
    return model_partition_id_md5.hexdigest()
