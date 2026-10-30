import asyncio
from typing import override

from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.application.postbox.abc.inbox import Inbox
from worker.domain.context.model_pass_context import ModelPassId
from worker.domain.model.model_pass import ModelPassError, ModelPassResult


class DefaultModelPassResultRetriever(ModelPassResultRetriever):
    def __init__(
        self,
        result_inbox: Inbox[ModelPassId, ModelPassResult],
        error_inbox: Inbox[ModelPassId, ModelPassError],
    ):
        self._result_inbox = result_inbox
        self._error_inbox = error_inbox

    @override
    async def retrieve_result(
        self,
        request_id: ModelPassId,
    ) -> ModelPassResult:
        result_task = asyncio.create_task(self._result_inbox.extract_by_key(request_id))
        error_task = asyncio.create_task(self._error_inbox.extract_by_key(request_id))

        try:
            await asyncio.wait(
                (result_task, error_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if error_task.done():
                error = error_task.result()
                raise RuntimeError(f"Model pass {request_id} failed: {error.message}")

            model_pass_result = result_task.result()
        finally:
            for task in (result_task, error_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(result_task, error_task, return_exceptions=True)

        return model_pass_result
