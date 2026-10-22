from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from integration.service_instance.service_instance import ServiceInstance
from model_manager.application.ports.inbound.model_partitioner import ModelPartitioner
from model_manager.application.ports.inbound.model_registry import ModelRegistry
from model_manager.application.ports.inbound.model_variant_registry import (
    ModelVariantRegistry,
)


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[
        tuple[ModelRegistry, ModelVariantRegistry, ModelPartitioner]
    ]: ...
