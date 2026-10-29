from typing import override

from worker.application.contribution.abc.contribution_pipeline import (
    ContributionPipeline,
)
from worker.application.contribution.abc.outbox import Outbox
from worker.application.ports.inbound.sink.model_pass_result_contribution_sink import (
    ModelPassResultContributionSink,
)
from worker.domain.model_pass.model_pass_error import ModelPassError
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class DefaultModelPassResultContributionSink(ModelPassResultContributionSink):
    def __init__(
        self,
        contribution_pipeline: ContributionPipeline[ModelPassResultContribution],
        error_outbox: Outbox[ModelPassError],
    ) -> None:
        self._contribution_pipeline = contribution_pipeline
        self._error_outbox = error_outbox

    @override
    async def sink_model_pass_contribution(
        self,
        contribution: ModelPassResultContribution,
    ) -> None:
        handle = await self._contribution_pipeline.submit(contribution)

        await handle.future

    @override
    async def notify_error(self, error: ModelPassError) -> None:
        await self._error_outbox.deposit(error)
