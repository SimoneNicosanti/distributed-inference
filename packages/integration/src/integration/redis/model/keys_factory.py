import asyncio
import hashlib

from integration.shared.id_json import model_id_to_json
from shared.model.model import ModelId


async def build_model_key(model_id: ModelId) -> str:
    md5_value = await asyncio.to_thread(
        hashlib.md5, model_id_to_json(model_id).encode("utf-8")
    )
    return f"model:{md5_value.hexdigest()}"
