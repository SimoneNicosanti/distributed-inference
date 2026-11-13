from fastapi import APIRouter, HTTPException, status

from integration.optimizer.http.optimizer_http_paths import OPTIMIZE_PATH
from integration.optimizer.http.optimizer_http_schema import (
    OptimizeRequest,
    OptimizeResponse,
    validate_solution_for_problem,
)
from optimizer.adapters.inbound.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper,
)
from optimizer.adapters.inbound.mappers.optimization_solution_mapper import (
    OptimizationSolutionMapper,
)
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController


def build_optimizer_router(optimizer_controller: OptimizerController) -> APIRouter:
    router = APIRouter(tags=["optimizer"])

    @router.post(OPTIMIZE_PATH, response_model=OptimizeResponse)
    async def optimize(request: OptimizeRequest) -> OptimizeResponse:
        try:
            problem = OptimizationProblemMapper.from_dto(request.problem)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        solution = await optimizer_controller.optimize(problem)
        result = OptimizationSolutionMapper.to_dto(solution)
        validate_solution_for_problem(request.problem, result)
        return OptimizeResponse(solution=result)

    return router
