import asyncio
import hashlib
import json
from typing import override

from model_manager.application.abc.model_variant_id_factory import ModelVariantIdFactory
from model_manager.domain.model_variant import ModelVariantInfo
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class DefaultModelVariantIdFactory(ModelVariantIdFactory):
    @staticmethod
    @override
    async def build_model_variant_id(
        model_id: ModelId, model_variant_info: ModelVariantInfo
    ) -> ModelVariantId:

        digest = await asyncio.to_thread(
            DefaultModelVariantIdFactory._compute_digest, model_variant_info
        )

        variant_tag = (
            f"{model_variant_info.format.value}-"
            f"{model_variant_info.precision.value}-"
            f"{model_variant_info.quantization.value}-"
            f"{digest}"
        )

        return ModelVariantId(model_id=model_id, variant_tag=variant_tag)

    @staticmethod
    def _compute_digest(model_variant_info: ModelVariantInfo) -> str:
        payload = {
            "format": model_variant_info.format.value,
            "precision": model_variant_info.precision.value,
            "quantization": model_variant_info.quantization.value,
            "inputs_info": {
                name: input_info.shape
                for name, input_info in sorted(model_variant_info.inputs_info.items())
            },
            "dynamic_dimension_values": {
                name: sorted(values.values)
                for name, values in sorted(
                    model_variant_info.dynamic_dimension_values.items()
                )
            },
        }

        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        return digest
