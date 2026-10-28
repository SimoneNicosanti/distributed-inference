import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import override

from integration.shared.id_json import model_id_to_json, model_variant_id_to_json
from shared.model.model_variant import ModelVariantId


class ProfiledModelVariantKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_profiled_model_variant_dict_name(
        model_variant_id: ModelVariantId,
    ) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_profiled_model_variant_key(
        model_variant_id: ModelVariantId,
    ) -> str: ...

    @staticmethod
    def build_profiled_model_variant_ids_key() -> str:
        return "profiled-model-variant:ids"


class HashProfiledModelVariantKeyFactory(ProfiledModelVariantKeyFactory):
    @staticmethod
    @override
    async def build_profiled_model_variant_dict_name(
        model_variant_id: ModelVariantId,
    ) -> str:
        model_id_md5_value = await asyncio.to_thread(
            hashlib.md5, model_id_to_json(model_variant_id.model_id).encode("utf-8")
        )

        return f"profiled-model-variant:{model_id_md5_value.hexdigest()}"

    @staticmethod
    @override
    async def build_profiled_model_variant_key(model_variant_id: ModelVariantId) -> str:
        model_variant_id_md5_value = await asyncio.to_thread(
            hashlib.md5, model_variant_id_to_json(model_variant_id).encode("utf-8")
        )
        return model_variant_id_md5_value.hexdigest()


class StringProfiledModelVariantKeyFactory(ProfiledModelVariantKeyFactory):
    @staticmethod
    @override
    async def build_profiled_model_variant_dict_name(
        model_variant_id: ModelVariantId,
    ) -> str:

        return f"profiled-model-variant:{model_variant_id.model_id.model_name}"

    @staticmethod
    @override
    async def build_profiled_model_variant_key(model_variant_id: ModelVariantId) -> str:
        return model_variant_id.variant_tag
