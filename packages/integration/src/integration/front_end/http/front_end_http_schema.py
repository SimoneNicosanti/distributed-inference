from pydantic import BaseModel

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId


class RunInferenceRequest(BaseModel):
    flow_id: FlowId
    bundle: TensorBundleDto


class RunInferenceResponse(BaseModel):
    bundle: TensorBundleDto


class LoadModelVariantResponse(BaseModel):
    model_variant_id: ModelVariantId
    artifact_ref: ArtifactRef
