from fastapi import APIRouter, HTTPException, status

from controller.adapters.inbound.mappers.optimization_request_mapper import (
    OptimizationRequestMapper,
)
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)


def build_optimization_manager_router(
    optimization_manager: OptimizationManager,
) -> APIRouter:
    router = APIRouter(tags=["controller-optimization"])

    @router.post(
        path=OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def optimize_inference_distribution(
        request: OptimizeInferenceDistributionRequest,
    ) -> None:
        try:
            requests = tuple(
                OptimizationRequestMapper.to_optimization_request(request_dto)
                for request_dto in request.requests
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        await optimization_manager.optimize_inference_distribution(
            {request.flow_id: request for request in requests}
        )

    return router
