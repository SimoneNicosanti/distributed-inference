from pydantic import BaseModel, ConfigDict

from shared.model.model import ModelId
from shared.model.model_variant import (
    AccuracyMetric,
    DynamicShape,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
    StaticShape,
)


class ModelVariantInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    static_shapes: list[StaticShape]
    dynamic_shapes: list[DynamicShape]

    ## TODO: Add check for shapes: no shape can be declared twice in the same or different shape group
    ## TODO: Add check for coherence between model type and architecture info


class ModelVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    model_variant_info: ModelVariantInfo

    @property
    def model_id(self) -> ModelId:
        return self.model_variant_id.model_id
