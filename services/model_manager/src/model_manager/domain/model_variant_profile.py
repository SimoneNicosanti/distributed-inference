from pydantic import BaseModel, ConfigDict

from model_manager.domain.model_variant_graph import ModelVariantGraph


class ModelVariantProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_graph: ModelVariantGraph
