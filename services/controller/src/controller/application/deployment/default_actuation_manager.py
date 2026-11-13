import asyncio
import logging
from contextlib import suppress
from typing import override

from controller.application.deployment.abc.actuation_manager import ActuationManager
from controller.application.deployment.abc.model_partition_build_manager import (
    ModelPartitionBuildManager,
)
from controller.application.deployment.abc.plan_mappings_builder import (
    PlanMappingsBuilder,
)
from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.abc.worker_plan_three_phase_dispatcher import (
    PlanThreePhaseDispatcher,
)
from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from lifecycle.async_lifecycle import AsyncLifecycle

logger = logging.getLogger(__name__)


class DefaultActuationManager(ActuationManager, AsyncLifecycle):
    def __init__(
        self,
        solution_observer: SolutionObserver,
        model_partition_build_manager: ModelPartitionBuildManager,
        plan_mappings_builder: PlanMappingsBuilder,
        plan_dispatcher: PlanThreePhaseDispatcher,
    ) -> None:
        self._solution_observer = solution_observer

        self._model_partition_build_manager = model_partition_build_manager
        self._plan_mappings_builder = plan_mappings_builder

        self._plan_dispatcher = plan_dispatcher

        self._observer_task: asyncio.Task[None] | None = None

    @override
    async def actuate_solution(self, solution: EnrichedOptimizationSolution) -> None:
        optimization_solution = solution.solution

        await self._model_partition_build_manager.build_model_partitions_from_solution(
            optimization_solution
        )

        (
            worker_plans_by_worker_id,
            flow_plans_by_flow_id,
        ) = await self._plan_mappings_builder.build_plan_mappings(solution)

        await self._plan_dispatcher.dispatch_plans(
            flow_plans_by_flow_id, worker_plans_by_worker_id
        )

    ## TODO: We should run the deploy in a different task and cancel it if a new solution arrives
    async def _observation_loop(self) -> None:
        ## TODO: Fix this try-except block
        while True:
            try:
                solution = await self._solution_observer.observe_new_solution()
                await self.actuate_solution(solution)
            except Exception:
                logger.exception("Solution observation or Solution actuation failed")

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
