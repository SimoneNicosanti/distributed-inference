from abc import ABC, abstractmethod


class ProfilingInferenceCoordinator(ABC):
    @abstractmethod
    def profile_model(self) -> None: ...
