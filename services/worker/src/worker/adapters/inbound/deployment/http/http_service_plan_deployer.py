from fastapi import APIRouter, HTTPException, status
from pydantic import PositiveInt

from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareWorkerPlanRequest,
)
from shared.plan.plan_version import PlanVersion
from worker.adapters.inbound.deployment.http.service_plan_mapper import (
    worker_plan_from_dto,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.plan.worker_plan import WorkerPlan


def build_worker_plan_deployment_router(
    worker_plan_deployer: PlanDeployer[WorkerPlan],
) -> APIRouter:
    router = APIRouter(tags=["worker-deployment"])
    prepared_versions: set[PlanVersion] = set()

    @router.put(
        path=PREPARE_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def prepare_worker_plan(
        version_number: PositiveInt,
        request: PrepareWorkerPlanRequest,
    ) -> None:
        try:
            plan = worker_plan_from_dto(request.worker_plan)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        plan_version = PlanVersion(version_number=version_number)
        if plan.plan_version != plan_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Path and worker plan versions do not match",
            )

        await worker_plan_deployer.prepare_plan(plan)
        prepared_versions.add(plan_version)

    @router.put(
        path=APPLY_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def apply_worker_plan(version_number: PositiveInt) -> None:
        plan_version = PlanVersion(version_number=version_number)
        if plan_version not in prepared_versions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Worker plan version was not prepared",
            )

        await worker_plan_deployer.apply_plan(plan_version)

    return router
