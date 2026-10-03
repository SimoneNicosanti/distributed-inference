import asyncio
import hashlib

from shared.model.model import ModelId


async def build_model_key(model_id: ModelId) -> str:
    md5_value = await asyncio.to_thread(
        hashlib.md5, model_id.model_dump_json().encode("utf-8")
    )
    return f"model:{md5_value.hexdigest()}"
