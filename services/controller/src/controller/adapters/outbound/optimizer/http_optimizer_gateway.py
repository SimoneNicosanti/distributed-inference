from typing import override

import httpx

from controller.adapters.outbound.optimizer.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper,
)
from controller.adapters.outbound.optimizer.mappers.optimization_solution_mapper import (
    OptimizationSolutionMapper,
)
from controller.application.ports.outbound.optimization.optimizer_gateway import (
    OptimizerGateway,
)
from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem,
)
from controller.domain.optimization.solution.optimization_solution import (
    OptimizationSolution,
)
from integration.optimizer.http.optimizer_http_paths import OPTIMIZE_PATH
from integration.optimizer.http.optimizer_http_schema import (
    OptimizeRequest,
    OptimizeResponse,
    validate_solution_for_problem,
)


class HttpOptimizerGateway(OptimizerGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        timeout_s: float = 300.0,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    @override
    async def optimize(self, problem: OptimizationProblem) -> OptimizationSolution:
        request = OptimizeRequest(problem=OptimizationProblemMapper.to_dto(problem))
        response = await self._http_client.post(
            f"{self._base_url}{OPTIMIZE_PATH}",
            json=request.model_dump(mode="json"),
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        result = OptimizeResponse.model_validate(response.json())
        validate_solution_for_problem(request.problem, result.solution)
        return OptimizationSolutionMapper.from_dto(result.solution)
