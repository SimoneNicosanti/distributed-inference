import asyncio

from fastapi import APIRouter, HTTPException, status
from pydantic import PositiveInt, ValidationError

from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareServicePlanRequest,
)
from shared.plan.plan_version import PlanVersion
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.service_plan import ServicePlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


def build_service_plan_deployment_router(
    worker_plan_deployer: PlanDeployer[WorkerPlan],
    ingress_plan_deployer: PlanDeployer[IngressPlan] | None,
) -> APIRouter:
    router = APIRouter(tags=["worker-deployment"])
    ingress_presence_by_version: dict[PlanVersion, bool] = {}

    @router.put(
        path=PREPARE_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def prepare_service_plan(
        version_number: PositiveInt,
        request: PrepareServicePlanRequest,
    ) -> None:
        try:
            plan = ServicePlan.model_validate(request.service_plan.model_dump())
        except ValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=error.errors(),
            ) from error

        plan_version = PlanVersion(version_number=version_number)
        if plan.worker_plan.plan_version != plan_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Path and service plan versions do not match",
            )

        has_ingress_plan = plan.ingress_plan is not None
        if has_ingress_plan and ingress_plan_deployer is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Service plan requires an unavailable ingress capability",
            )

        previous_ingress_presence = ingress_presence_by_version.get(plan_version)
        if (
            previous_ingress_presence is not None
            and previous_ingress_presence != has_ingress_plan
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Service plan version was prepared with different capabilities",
            )

        if plan.ingress_plan is not None and ingress_plan_deployer is not None:
            await asyncio.gather(
                worker_plan_deployer.prepare_plan(plan.worker_plan),
                ingress_plan_deployer.prepare_plan(plan.ingress_plan),
            )
        else:
            await worker_plan_deployer.prepare_plan(plan.worker_plan)

        ingress_presence_by_version[plan_version] = has_ingress_plan

    @router.put(
        path=APPLY_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def apply_service_plan(version_number: PositiveInt) -> None:
        plan_version = PlanVersion(version_number=version_number)
        if ingress_presence_by_version.get(plan_version, False):
            if ingress_plan_deployer is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Prepared ingress plan cannot be applied",
                )
            await asyncio.gather(
                worker_plan_deployer.apply_plan(plan_version),
                ingress_plan_deployer.apply_plan(plan_version),
            )
        else:
            await worker_plan_deployer.apply_plan(plan_version)

    return router
