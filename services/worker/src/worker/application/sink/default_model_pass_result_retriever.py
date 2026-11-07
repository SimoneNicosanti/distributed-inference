from typing import override

from shared.context.model_pass_context import ModelPassId
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.application.postbox.abc.inbox import KeyedInbox
from worker.domain.model.model_pass import (
    ModelPassError,
    ModelPassOutcome,
    ModelPassResult,
)


class DefaultModelPassResultRetriever(ModelPassResultRetriever):
    def __init__(
        self,
        outcome_inbox: KeyedInbox[ModelPassId, ModelPassOutcome],
    ):
        self._outcome_postbox = outcome_inbox

    @override
    async def retrieve_result(
        self,
        request_id: ModelPassId,
    ) -> ModelPassResult:

        outcome = await self._outcome_postbox.extract_by_key(request_id)

        if isinstance(outcome, ModelPassError):
            raise RuntimeError(f"Model pass {request_id} failed: {outcome.message}")

        return outcome
