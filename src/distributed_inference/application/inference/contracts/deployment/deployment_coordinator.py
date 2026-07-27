from abc import ABC, abstractmethod


class DeploymentCoordinator(ABC):
    @abstractmethod
    def apply_plan(
        self,
    ) -> None: ...
