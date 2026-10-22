from pydantic import BaseModel

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.flow.flow import FlowId


class RunInferenceRequest(BaseModel):
    flow_id: FlowId
    bundle: TensorBundleDto


class RunInferenceResponse(BaseModel):
    bundle: TensorBundleDto


class LoadModelVariantResponse(BaseModel):
    artifact_ref: ArtifactRef
