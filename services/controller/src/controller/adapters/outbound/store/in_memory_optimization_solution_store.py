import asyncio
from typing import override

from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.ports.outbound.store.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution


class InMemoryOptimizationSolutionStore(OptimizationSolutionStore, SolutionObserver):
    def __init__(self) -> None:
        self._current_solution: OptimizationSolution | None = None
        self._updates: asyncio.Queue[OptimizationSolution] = asyncio.Queue(maxsize=1)

    @override
    async def put_solution(self, solution: OptimizationSolution) -> None:
        self._current_solution = solution

        if self._updates.full():
            self._updates.get_nowait()
            self._updates.task_done()

        self._updates.put_nowait(solution)

    @override
    async def get_current_solution(self) -> OptimizationSolution | None:
        return self._current_solution

    @override
    async def observe_new_solution(self) -> OptimizationSolution:
        solution = await self._updates.get()
        self._updates.task_done()
        return solution
