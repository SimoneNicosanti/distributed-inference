from pydantic import BaseModel

from shared.model.model_variant import (
    AccuracyMetric,
    DynamicShape,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
    StaticShape,
)


class ModelVariantDto(BaseModel):
    id: ModelVariantId

    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    static_shapes: list[StaticShape]
    dynamic_shapes: list[DynamicShape]
