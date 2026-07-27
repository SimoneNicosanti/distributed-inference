from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServerPlan


class DeploymentCoordinator(ABC):
    @abstractmethod
    async def apply_plan(
        self,
        server_plan: ServerPlan,
    ) -> None: ...
