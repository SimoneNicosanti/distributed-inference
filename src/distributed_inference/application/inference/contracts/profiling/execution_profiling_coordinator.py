from abc import ABC, abstractmethod


class ExecutionProfilingCoordinator(ABC):
    @abstractmethod
    def profile_model(self) -> None: ...
