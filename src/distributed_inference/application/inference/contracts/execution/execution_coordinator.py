from abc import ABC, abstractmethod


class InferenceCoordinator(ABC):
    @abstractmethod
    def execute_stage(self) -> None: ...
