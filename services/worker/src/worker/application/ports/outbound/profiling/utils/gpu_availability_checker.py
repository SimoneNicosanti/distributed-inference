from abc import ABC, abstractmethod


class GPUAvailabilityChecker(ABC):
    @staticmethod
    @abstractmethod
    def check() -> bool: ...
