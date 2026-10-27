from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariantInfo,
)


class ModelVariantInfoMapper:
    @staticmethod
    def to_model_variant_info_from_dto(
        model_variant_info_dto: ModelVariantInfoDto,
    ) -> ModelVariantInfo:
        dto_info = model_variant_info_dto
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

        return model_variant_info
