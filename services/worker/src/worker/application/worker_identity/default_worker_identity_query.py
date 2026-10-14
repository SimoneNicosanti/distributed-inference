from typing import override

from shared.service.service import WorkerId
from worker.application.worker_identity.abc.worker_identity_query import (
    WorkerIdentityQuery,
)


class DefaultWorkerIdentityQuery(WorkerIdentityQuery):
    def __init__(self, worker_id: WorkerId) -> None:
        self._worker_id = worker_id

    @override
    async def get_worker_identity(self) -> WorkerId:
        return self._worker_id
