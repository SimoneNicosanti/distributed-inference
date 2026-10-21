import asyncio
from contextlib import suppress
from typing import override

from controller.application.deployment.abc.actuation_manager import ActuationManager
from controller.application.deployment.abc.service_plan_builder import (
    ServicePlanBuilder,
)
from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.builder.model_partition_id_builder import (
    ModelPartitionIdBuilder,
)
from controller.application.deployment.builder.plan_version_builder import (
    PlanVersionBuilder,
)
from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.domain.optimization.optimization_solution import OptimizationSolution
from lifecycle.async_lifecycle import AsyncLifecycle


class DefaultActuationManager(ActuationManager, AsyncLifecycle):
    def __init__(
        self,
        solution_observer: SolutionObserver,
        partition_builder_gateway: PartitionBuilderGateway,
        plan_version_builder: PlanVersionBuilder,
        service_plan_builder: ServicePlanBuilder,
        plan_share_gateway: PlanShareGateway,
    ) -> None:
        self._solution_observer = solution_observer

        self._partition_builder_gateway = partition_builder_gateway

        self._plan_version_builder = plan_version_builder
        self._service_plan_builder = service_plan_builder

        self._plan_share_gateway = plan_share_gateway

        self._observer_task: asyncio.Task[None] | None = None
        pass

    @override
    async def actuate_solution(self, solution: OptimizationSolution) -> None:

        model_partition_ids = (
            ModelPartitionIdBuilder.build_all_model_partition_ids_from_solution(
                solution,
            )
        )
        await self._partition_builder_gateway.build_all_partitions_by_ids(
            model_partition_ids,
        )

        plan_version = self._plan_version_builder.build_next_plan_version()

        service_plan_by_worker_id = (
            await self._service_plan_builder.build_all_plans_from_solution(
                plan_version,
                solution,
            )
        )

        await asyncio.gather(
            *[
                self._plan_share_gateway.send_prepare_plan_message(worker_id, plan)
                for worker_id, plan in service_plan_by_worker_id.items()
            ]
        )

        await asyncio.gather(
            *[
                self._plan_share_gateway.send_apply_plan_message(
                    worker_id, plan_version
                )
                for worker_id, plan in service_plan_by_worker_id.items()
            ]
        )

    ## We should run the deploy in a different task and cancel it if a new solution arrives
    async def _observation_loop(self) -> None:
        while True:
            solution = await self._solution_observer.observe_new_solution()
            await self.actuate_solution(solution)

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
