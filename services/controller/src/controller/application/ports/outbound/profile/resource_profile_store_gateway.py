from abc import ABC, abstractmethod
from collections.abc import Iterable

from controller.domain.profile.resource_profile import WorkerResourceProfile
from shared.service.service import WorkerId


class ResourceProfileStoreGateway(ABC):
    @abstractmethod
    async def get_resource_profiles_for_workers(
        self, worker_ids: Iterable[WorkerId]
    ) -> list[WorkerResourceProfile]: ...
