import asyncio

from fastapi import APIRouter, HTTPException, status
from pydantic import PositiveInt

from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareServicePlanRequest,
)
from shared.plan.plan_version import PlanVersion
from worker.adapters.inbound.deployment.http.service_plan_mapper import (
    service_plan_from_dto,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.sink.sink_plan import SinkPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


def build_service_plan_deployment_router(
    worker_plan_deployer: PlanDeployer[WorkerPlan],
    ingress_plan_deployer: PlanDeployer[IngressPlan] | None,
    sink_plan_deployer: PlanDeployer[SinkPlan] | None,
) -> APIRouter:
    router = APIRouter(tags=["worker-deployment"])
    role_presence_by_version: dict[PlanVersion, tuple[bool, bool]] = {}

    @router.put(
        path=PREPARE_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def prepare_service_plan(
        version_number: PositiveInt,
        request: PrepareServicePlanRequest,
    ) -> None:
        try:
            plan = service_plan_from_dto(request.service_plan)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        plan_version = PlanVersion(version_number=version_number)
        if plan.worker_plan.plan_version != plan_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Path and service plan versions do not match",
            )

        has_ingress_plan = plan.ingress_plan is not None
        has_sink_plan = plan.sink_plan is not None
        if has_ingress_plan and ingress_plan_deployer is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Service plan requires an unavailable ingress capability",
            )
        if has_sink_plan and sink_plan_deployer is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Service plan requires an unavailable sink capability",
            )

        role_presence = (has_ingress_plan, has_sink_plan)
        previous_role_presence = role_presence_by_version.get(plan_version)
        if (
            previous_role_presence is not None
            and previous_role_presence != role_presence
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Service plan version was prepared with different capabilities",
            )

        prepare_operations = [worker_plan_deployer.prepare_plan(plan.worker_plan)]
        if plan.ingress_plan is not None and ingress_plan_deployer is not None:
            prepare_operations.append(
                ingress_plan_deployer.prepare_plan(plan.ingress_plan)
            )
        if plan.sink_plan is not None and sink_plan_deployer is not None:
            prepare_operations.append(sink_plan_deployer.prepare_plan(plan.sink_plan))

        await asyncio.gather(*prepare_operations)
        role_presence_by_version[plan_version] = role_presence

    @router.put(
        path=APPLY_SERVICE_PLAN_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def apply_service_plan(version_number: PositiveInt) -> None:
        plan_version = PlanVersion(version_number=version_number)
        role_presence = role_presence_by_version.get(plan_version)
        if role_presence is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Service plan version was not prepared",
            )

        has_ingress_plan, has_sink_plan = role_presence
        apply_operations = [worker_plan_deployer.apply_plan(plan_version)]
        if has_ingress_plan:
            if ingress_plan_deployer is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Prepared ingress plan cannot be applied",
                )
            apply_operations.append(ingress_plan_deployer.apply_plan(plan_version))
        if has_sink_plan:
            if sink_plan_deployer is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Prepared sink plan cannot be applied",
                )
            apply_operations.append(sink_plan_deployer.apply_plan(plan_version))

        await asyncio.gather(*apply_operations)

    return router
