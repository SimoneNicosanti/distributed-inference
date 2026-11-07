from typing import override

from shared.context.model_pass_context import ModelPassId
from worker.application.ports.inbound.sink.model_pass_outcome_retriever import (
    ModelPassOutcomeRetriever,
)
from worker.application.postbox.abc.inbox import KeyedInbox
from worker.domain.model.model_pass import (
    ModelPassOutcome,
)


class DefaultModelPassResultRetriever(ModelPassOutcomeRetriever):
    def __init__(
        self,
        inbox: KeyedInbox[ModelPassId, ModelPassOutcome],
    ):
        self._inbox = inbox

    @override
    async def retrieve_outcome(
        self,
        request_id: ModelPassId,
    ) -> ModelPassOutcome:

        outcome = await self._inbox.extract_by_key(request_id)

        return outcome
