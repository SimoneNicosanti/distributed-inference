from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from integration.dto.model_pass_error.model_pass_error_dto import ModelPassErrorDto
from integration.worker.failure.http_failure_schema import (
    NOTIFY_FAILURE_PATH,
)
from worker.adapters.context_mapper import model_pass_context_from_dto
from worker.application.ports.inbound.failure_rcv.failure_reciever import (
    FailureReceiver,
)
from worker.domain.model.model_pass import ModelPassError


def build_failure_receiver_router(receiver: FailureReceiver) -> APIRouter:
    router = APIRouter(tags=["worker-failure"])

    @router.post(path=NOTIFY_FAILURE_PATH, status_code=status.HTTP_204_NO_CONTENT)
    async def notify_failure(model_pass_id: UUID, request: ModelPassErrorDto) -> None:
        if model_pass_id != request.context.model_pass_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Path and model pass IDs do not match",
            )
        await receiver.notify_failure(
            ModelPassError(
                context=model_pass_context_from_dto(request.context),
                message=request.message,
            )
        )

    return router
