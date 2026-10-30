from pydantic import BaseModel, ConfigDict

from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto


class PrepareWorkerPlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_plan: WorkerPlanDto
