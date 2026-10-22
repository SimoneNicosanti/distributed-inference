from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from integration.dto.optimization.optimization_dto import OptimizationRequestDto


class OptimizeInferenceDistributionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: Annotated[tuple[OptimizationRequestDto, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_flow_ids(self) -> Self:
        flow_ids = tuple(request.flow_id for request in self.requests)
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("Optimization requests must have unique flow IDs")
        return self
