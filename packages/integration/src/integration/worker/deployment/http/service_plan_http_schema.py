from pydantic import BaseModel, ConfigDict

from integration.dto.service_plan.service_plan_dto import WorkerPlanDto


class PrepareWorkerPlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_plan: WorkerPlanDto
