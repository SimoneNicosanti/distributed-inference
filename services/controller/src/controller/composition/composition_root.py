from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from integration.service_instance.service_instance import ServiceInstance


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[OptimizationManager]: ...
