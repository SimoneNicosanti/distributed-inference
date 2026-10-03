from abc import ABC, abstractmethod

from model_manager.domain.model import Model


class ModelRegistry(ABC):
    @abstractmethod
    async def register_model(self, model: Model) -> None: ...
