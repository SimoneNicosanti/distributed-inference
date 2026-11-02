from typing import override

import httpx

from front_end.application.ports.outbound.optimization.optimization_gateway import (
    OptimizationGateway,
)
from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)
from integration.dto.optimization.optimization_dto import OptimizationRequestDto


class HttpOptimizationGateway(OptimizationGateway):
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
    async def optimize_inference_distribution(
        self, requests: tuple[OptimizationRequestDto, ...]
    ) -> None:
        request = OptimizeInferenceDistributionRequest(requests=requests)
        response = await self._http_client.post(
            f"{self._base_url}{OPTIMIZE_INFERENCE_DISTRIBUTION_PATH}",
            json=request.model_dump(mode="json"),
            timeout=self._timeout_s,
        )
        response.raise_for_status()
