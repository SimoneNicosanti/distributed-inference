from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from front_end.application.ports.inbound.inference_runner import (
    InferenceCoordinator,
)
from front_end.application.ports.inbound.model_loader import ModelLoader
from front_end.application.ports.outbound.optimization.optimization_gateway import (
    OptimizationGateway,
)
from front_end.application.ports.outbound.inference.local_worker_identity_gateway import (
    LocalWorkerIdentityGateway,
)


@dataclass(frozen=True)
class FrontEndComposition:
    inference_runner: InferenceCoordinator
    worker_identity_gateway: LocalWorkerIdentityGateway
    model_loader: ModelLoader
    optimization_gateway: OptimizationGateway


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(self) -> AsyncGenerator[FrontEndComposition]: ...
