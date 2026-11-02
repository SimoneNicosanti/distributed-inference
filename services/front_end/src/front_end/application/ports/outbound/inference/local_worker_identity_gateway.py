from abc import ABC, abstractmethod

from shared.service.service import WorkerId


class LocalWorkerIdentityGateway(ABC):
    @abstractmethod
    async def get_local_worker_identity(self) -> WorkerId: ...
