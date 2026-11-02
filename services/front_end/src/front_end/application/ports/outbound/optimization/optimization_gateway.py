from abc import ABC, abstractmethod

from integration.dto.optimization.optimization_dto import OptimizationRequestDto


class OptimizationGateway(ABC):
    @abstractmethod
    async def optimize_inference_distribution(
        self, requests: tuple[OptimizationRequestDto, ...]
    ) -> None: ...
