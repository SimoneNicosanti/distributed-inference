import asyncio
from typing import override

from shared.context.model_pass_context import ModelPassId
from worker.application.contribution.abc.contribution_assembler import (
    ContributionAssembler,
)
from worker.application.contribution.abc.inbox import Inbox
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.application.sink.model_pass_result_collection import (
    ModelPassResultCollection,
)
from worker.domain.model_pass.model_pass_error import ModelPassError
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class DefaultModelPassResultRetriever(ModelPassResultRetriever):
    def __init__(
        self,
        inbox: Inbox[ModelPassId, ModelPassResultCollection],
        error_inbox: Inbox[ModelPassId, ModelPassError],
        assembler: ContributionAssembler[ModelPassResultContribution, ModelPassResult],
    ):
        self._inbox = inbox
        self._error_inbox = error_inbox
        self._assembler = assembler

    @override
    async def retrieve_result(
        self,
        request_id: ModelPassId,
    ) -> ModelPassResult:
        result_task = asyncio.create_task(self._inbox.extract_by_key(request_id))
        error_task = asyncio.create_task(self._error_inbox.extract_by_key(request_id))

        try:
            await asyncio.wait(
                (result_task, error_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if error_task.done():
                error = error_task.result()
                raise RuntimeError(f"Model pass {request_id} failed: {error.message}")

            collection = result_task.result()
        finally:
            for task in (result_task, error_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(result_task, error_task, return_exceptions=True)

        contributions = list(collection.contributions)
        model_pass_result = await self._assembler.assemble(contributions)
        return model_pass_result
