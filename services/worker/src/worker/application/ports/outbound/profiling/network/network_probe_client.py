from abc import ABC, abstractmethod

from worker.domain.directory.worker_instance import WorkerInstance
from worker.domain.profiling.network_profile import ConnectionInfo


class NetworkProbeClient(ABC):
    @abstractmethod
    async def probe_connection(
        self, worker_instance: WorkerInstance
    ) -> ConnectionInfo: ...
