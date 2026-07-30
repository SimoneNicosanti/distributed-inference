from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# class InferenceRequestId(BaseModel):
#     model_config = ConfigDict(frozen=True)
#     id: UUID = Field(default_factory=uuid4)

## An inference run represents the invocation of multiple models
# class InferenceRunContext(BaseModel):
#     model_config = ConfigDict(frozen=True)
#     inference_request_id: InferenceRequestId
#     inference_run_id: UUID = Field(default_factory=uuid4)


## This is the context id for model invocation.
## The user can call for a model invocation multiple times;
## this is what is used to distiguish between invocations
class ModelInvocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    # inference_run_context: InferenceRunContext
    model_invocation_id: UUID = Field(default_factory=uuid4)
