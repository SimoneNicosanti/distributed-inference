from fastapi import APIRouter

from integration.worker.worker_identity.http_worker_identity_query_paths import (
    WORKER_IDENTITY_PATH,
)
from shared.service.service import WorkerId
from worker.application.worker_identity.abc.worker_identity_query import (
    WorkerIdentityQuery,
)


def build_worker_identity_query_router(
    worker_identity_query: WorkerIdentityQuery,
) -> APIRouter:

    router = APIRouter()

    @router.get(path=WORKER_IDENTITY_PATH, response_model=WorkerId)
    async def get_worker_identity() -> WorkerId:
        return await worker_identity_query.get_worker_identity()

    return router
