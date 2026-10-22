from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.flow.flow import FlowId
from shared.service.service import ResultSinkId

# @dataclass(frozen=True, slots=True)
# class InferenceRequestId:
#     id: UUID = field(default_factory=uuid4)

## An inference run represents the invocation of multiple models
# @dataclass(frozen=True, slots=True)
# class InferenceRunContext:
#     inference_request_id: InferenceRequestId
#     inference_run_id: UUID = field(default_factory=uuid4)

type ModelInvocationId = UUID


## This is the context id for model invocation.
## The user can call for a model multiple times;
## this is what is used to distiguish between invocations
@dataclass(frozen=True, slots=True)
class ModelInvocationContext:
    # inference_run_context: InferenceRunContext
    flow_id: FlowId
    result_sink: ResultSinkId
    model_invocation_id: ModelInvocationId = field(default_factory=uuid4)
