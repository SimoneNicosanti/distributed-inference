from abc import ABC, abstractmethod
from collections.abc import Iterable

from controller.domain.profile.network_topology import (
    NetworkConnection,
)
from shared.service.service import WorkerId


class NetworkProfileStoreGateway(ABC):
    @abstractmethod
    async def get_network_connections_by_workers(
        self, worker_ids: Iterable[WorkerId]
    ) -> list[NetworkConnection]: ...
