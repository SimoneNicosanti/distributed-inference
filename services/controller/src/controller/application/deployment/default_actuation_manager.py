import asyncio
from contextlib import suppress
from typing import override

from controller.application.deployment.abc.actuation_manager import ActuationManager
from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.domain.optimization.optimization_solution import OptimizationSolution
from lifecycle.async_lifecycle import AsyncLifecycle


class DefaultActuationManager(ActuationManager, AsyncLifecycle):
    def __init__(
        self,
        solution_observer: SolutionObserver,
    ) -> None:
        self._solution_observer = solution_observer
        self._observer_task: asyncio.Task[None] | None = None
        pass

    @override
    async def deploy_solution(self, solution: OptimizationSolution) -> None:
        ## TODO: implement

        ## Building of sub models:
        ## 1. Check how components have been build
        ## 2. Ask for component build to model manager

        ## Building of plan:
        ## 1. For all involved workers
        ## 2. Build corresponding plan

        ## Plan preparation message:
        ## 3. Send plan to workers

        ## Plan application message:
        ## 4. Send apply message to all workers
        pass

    async def _observation_loop(self) -> None:
        while True:
            solution = await self._solution_observer.observe_new_solution()
            await self.deploy_solution(solution)

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
