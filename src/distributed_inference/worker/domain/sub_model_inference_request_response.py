from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.plan import InferencePlanVersion


## This is the input of the sub-model once everything for the sub-model has been gathered from predecessors
class SubModelInferenceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    inference_plan_version: InferencePlanVersion
    pass


## This is the output of the sub-model as returned to the caller
class SubModelInferenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass
