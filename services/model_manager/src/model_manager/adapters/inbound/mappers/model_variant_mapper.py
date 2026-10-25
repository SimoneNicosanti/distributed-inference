from integration.dto.model.model_variant_dto import ModelVariantDto
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)


class ModelVariantMapper:
    @staticmethod
    def to_model_variant_from_dto(model_variant_dto: ModelVariantDto) -> ModelVariant:
        dto_info = model_variant_dto.info
        model_variant_info = ModelVariantInfo(
            precision=dto_info.precision,
            quantization=dto_info.quantization,
            accuracies=tuple(dto_info.accuracies),
            format=dto_info.format,
            inputs_info={
                name: InputInfo(shape=input_info.shape)
                for name, input_info in dto_info.inputs_info.items()
            },
            dynamic_dimension_values={
                name: DynamicShapeValues(values=values.values)
                for name, values in dto_info.dynamic_dimension_values.items()
            },
        )

        model_variant = ModelVariant(
            model_variant_id=model_variant_dto.id,
            model_variant_info=model_variant_info,
        )
        return model_variant
