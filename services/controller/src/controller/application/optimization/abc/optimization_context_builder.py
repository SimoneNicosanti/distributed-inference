from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_context import OptimizationContext


class OptimizationContextBuilder(ABC):
    @abstractmethod
    async def build_optimization_context(self) -> OptimizationContext: ...
