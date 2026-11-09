import asyncio
from collections.abc import Awaitable, Mapping
from typing import override

from controller.application.deployment.abc.worker_plan_three_phase_dispatcher import (
    PlanThreePhaseDispatcher,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class DefaultPlanThreePhaseDispatcher(PlanThreePhaseDispatcher):
    def __init__(
        self,
        plan_share_gateway: PlanShareGateway,
        flow_plan_publisher: FlowPlanPublisher,
    ) -> None:
        super().__init__()
        self._plan_share_gateway: PlanShareGateway = plan_share_gateway
        self._flow_plan_publisher: FlowPlanPublisher = flow_plan_publisher

    @override
    async def dispatch_plans(
        self,
        flow_plans: dict[FlowId, FlowPlan],
        worker_plans: dict[WorkerId, WorkerPlan],
    ) -> None:

        ## TODO: Here we should handle rollback

        ## Prepare plan message
        await self._run_worker_phase(
            "deploy",
            {
                (
                    worker_id,
                    plan.plan_version,
                ): self._plan_share_gateway.send_prepare_plan_message(worker_id, plan)
                for worker_id, plan in worker_plans.items()
            },
        )

        ## Apply plan message
        await self._run_worker_phase(
            "apply",
            {
                (
                    worker_id,
                    plan.plan_version,
                ): self._plan_share_gateway.send_apply_plan_message(
                    worker_id, plan.plan_version
                )
                for worker_id, plan in worker_plans.items()
            },
        )

        ## Pubblicazione del piano dei flussi
        try:
            await self._flow_plan_publisher.publish_flow_plans(flow_plans)
        except Exception as error:
            error.add_note("Flow plan publication failed")
            raise

    async def _run_worker_phase(
        self,
        phase: str,
        operations: Mapping[tuple[WorkerId, PlanVersion], Awaitable[None]],
    ) -> None:
        results = await asyncio.gather(*operations.values(), return_exceptions=True)
        failures: list[Exception] = []
        for deploy_info, result in zip(operations, results, strict=True):
            worker_id, plan_version = deploy_info
            if isinstance(result, BaseException):
                result.add_note(
                    f"Worker {worker_id.id}, phase {phase}, "
                    f"plan version {plan_version.version_number}"
                )
                if not isinstance(result, Exception):
                    raise result
                failures.append(result)
        if failures:
            raise ExceptionGroup(
                f"Worker {phase} failed",
                failures,
            )
