import asyncio
import hashlib

from integration.shared.id_json import model_id_to_json, model_variant_id_to_json
from shared.model.model_variant import ModelVariantId


async def build_profiled_model_variant_dict_name(
    model_variant_id: ModelVariantId,
) -> str:
    model_id_md5_value = await asyncio.to_thread(
        hashlib.md5, model_id_to_json(model_variant_id.model_id).encode("utf-8")
    )

    return f"profiled-model-variant:{model_id_md5_value.hexdigest()}"


async def build_profiled_model_variant_key(model_variant_id: ModelVariantId) -> str:
    model_variant_id_md5_value = await asyncio.to_thread(
        hashlib.md5, model_variant_id_to_json(model_variant_id).encode("utf-8")
    )
    return model_variant_id_md5_value.hexdigest()


def build_profiled_model_variant_ids_key() -> str:
    return "profiled-model-variant:ids"
