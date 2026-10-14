from abc import ABC, abstractmethod

from shared.service.service import WorkerId


class WorkerIdentityQuery(ABC):
    @abstractmethod
    async def get_worker_identity(self) -> WorkerId: ...
