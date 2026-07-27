from abc import ABC, abstractmethod


class ProfilingCoordinator(ABC):
    @abstractmethod
    def execute_profiling(self) -> None: ...
