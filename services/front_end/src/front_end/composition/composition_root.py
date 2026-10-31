from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from front_end.application.ports.inbound.inference_runner import (
    InferenceCoordinator,
    LocalWorkerIdentityQuery,
)
from front_end.application.ports.inbound.model_loader import ModelLoader


@dataclass(frozen=True)
class FrontEndComposition:
    inference_runner: InferenceCoordinator
    local_worker_identity_query: LocalWorkerIdentityQuery
    model_loader: ModelLoader


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(self) -> AsyncGenerator[FrontEndComposition]: ...
