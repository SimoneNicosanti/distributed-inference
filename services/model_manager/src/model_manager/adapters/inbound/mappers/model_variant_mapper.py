from integration.dto.model.model_variant_dto import ModelVariantDto
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)


class ModelVarianMapper:
    @staticmethod
    def to_model_variant_from_dto(model_variant_dto: ModelVariantDto) -> ModelVariant:
        model_variant_info = ModelVariantInfo(
            precision=model_variant_dto.precision,
            quantization=model_variant_dto.quantization,
            accuracies=tuple(model_variant_dto.accuracies),
            format=model_variant_dto.format,
            inputs_info={
                name: InputInfo(shape=info.shape)
                for name, info in model_variant_dto.inputs_info.items()
            },
            dynamic_dimension_values={
                name: DynamicShapeValues(values=values.values)
                for name, values in (model_variant_dto.dynamic_dimension_values.items())
            },
        )

        model_variant = ModelVariant(
            model_variant_id=model_variant_dto.id,
            model_variant_info=model_variant_info,
        )
        return model_variant
