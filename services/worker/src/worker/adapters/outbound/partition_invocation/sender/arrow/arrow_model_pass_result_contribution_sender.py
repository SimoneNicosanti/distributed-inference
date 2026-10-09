from worker.application.ports.outbound.partition_invocation.sender.contribution_sender import (
    ContributionSender,
)
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class ArrowModelPassResultContributionSender(
    ContributionSender[ModelPassResultContribution]
):
    def __init__(self) -> None:
        pass

    async def send(self, contribution: ModelPassResultContribution) -> None:
        pass
