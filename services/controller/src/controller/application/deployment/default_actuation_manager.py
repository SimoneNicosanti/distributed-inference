import asyncio
import logging
from collections.abc import Awaitable, Mapping
from contextlib import suppress
from typing import override

from controller.application.deployment.abc.actuation_manager import ActuationManager
from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.abc.solution_to_plan_converter import (
    SolutionToPlanConverter,
)
from controller.application.deployment.builder.model_partition_id_builder import (
    ModelPartitionIdBuilder,
)
from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.application.ports.outbound.plan_version.plan_version_factory import (
    PlanVersionFactory,
)
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from lifecycle.async_lifecycle import AsyncLifecycle
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId

logger = logging.getLogger(__name__)


class DefaultActuationManager(ActuationManager, AsyncLifecycle):
    def __init__(
        self,
        solution_observer: SolutionObserver,
        partition_builder_gateway: PartitionBuilderGateway,
        plan_version_factory: PlanVersionFactory,
        worker_plan_builder: SolutionToPlanConverter,
        plan_share_gateway: PlanShareGateway,
        flow_plan_publisher: FlowPlanPublisher,
    ) -> None:
        self._solution_observer = solution_observer

        self._partition_builder_gateway = partition_builder_gateway

        self._plan_version_factory = plan_version_factory
        self._worker_plan_builder = worker_plan_builder

        self._plan_share_gateway = plan_share_gateway
        self._flow_plan_publisher = flow_plan_publisher

        self._observer_task: asyncio.Task[None] | None = None

    @override
    async def actuate_solution(self, solution: EnrichedOptimizationSolution) -> None:
        """Publish after all workers apply; failures do not roll back completed work."""
        optimization_solution = solution.solution

        model_partition_ids = (
            ModelPartitionIdBuilder.build_all_model_partition_ids_from_solution(
                optimization_solution,
            )
        )
        await self._partition_builder_gateway.build_all_partitions_by_ids(
            model_partition_ids,
        )

        plan_version = await self._plan_version_factory.build_next_plan_version()

        (
            worker_plans_by_worker_id,
            flow_plans_by_flow_id,
        ) = await self._worker_plan_builder.convert_solution_to_plans(
            plan_version,
            solution,
        )

        await self._run_worker_phase(
            "deploy",
            plan_version,
            {
                worker_id: self._plan_share_gateway.send_prepare_plan_message(
                    worker_id, plan
                )
                for worker_id, plan in worker_plans_by_worker_id.items()
            },
        )

        await self._run_worker_phase(
            "apply",
            plan_version,
            {
                worker_id: self._plan_share_gateway.send_apply_plan_message(
                    worker_id, plan_version
                )
                for worker_id in worker_plans_by_worker_id
            },
        )

        try:
            await self._flow_plan_publisher.publish_flow_plans(flow_plans_by_flow_id)
        except Exception as error:
            error.add_note(
                f"Flow plan publication failed for plan version {plan_version.version_number}"
            )
            raise

    async def _run_worker_phase(
        self,
        phase: str,
        plan_version: PlanVersion,
        operations: Mapping[WorkerId, Awaitable[None]],
    ) -> None:
        results = await asyncio.gather(*operations.values(), return_exceptions=True)
        failures: list[Exception] = []
        for worker_id, result in zip(operations, results, strict=True):
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
                f"Worker {phase} failed for plan version {plan_version.version_number}",
                failures,
            )

    ## We should run the deploy in a different task and cancel it if a new solution arrives
    async def _observation_loop(self) -> None:
        while True:
            solution = await self._solution_observer.observe_new_solution()
            try:
                await self.actuate_solution(solution)
            except Exception:
                logger.exception("Solution actuation failed")

    @override
    async def start(self) -> None:
        if self._observer_task is not None:
            return
        self._observer_task = asyncio.create_task(self._observation_loop())
        pass

    @override
    async def stop(self) -> None:
        if self._observer_task is None:
            return
        self._observer_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._observer_task

        self._observer_task = None
