from abc import ABC, abstractmethod


class DeploymentCoordinator(ABC):
    @abstractmethod
    async def apply_plan(
        self,
    ) -> None: ...
