from fastapi import APIRouter

from integration.worker.worker_identity.http_worker_identity_query_paths import (
    WORKER_IDENTITY_PATH,
)
from integration.dto.worker_identity.worker_identity_dto import WorkerIdentityDto
from worker.application.worker_identity.abc.worker_identity_query import (
    WorkerIdentityQuery,
)


def build_worker_identity_query_router(
    worker_identity_query: WorkerIdentityQuery,
) -> APIRouter:

    router = APIRouter()

    @router.get(path=WORKER_IDENTITY_PATH, response_model=WorkerIdentityDto)
    async def get_worker_identity() -> WorkerIdentityDto:
        worker_id = await worker_identity_query.get_worker_identity()
        return WorkerIdentityDto(service_id=worker_id.service_id)

    return router
