from pydantic import BaseModel, ConfigDict

from integration.worker.deployment.dto.service_plan_dto import ServicePlanDto


class PrepareServicePlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_plan: ServicePlanDto
