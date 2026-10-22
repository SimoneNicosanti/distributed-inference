from pydantic import BaseModel, ConfigDict

from integration.dto.service_plan.service_plan_dto import ServicePlanDto


class PrepareServicePlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_plan: ServicePlanDto
