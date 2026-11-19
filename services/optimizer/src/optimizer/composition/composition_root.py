from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from integration.service_instance.service_instance import ServiceInstance
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[OptimizerController]: ...
