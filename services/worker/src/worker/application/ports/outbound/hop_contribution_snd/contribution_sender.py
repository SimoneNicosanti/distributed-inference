from abc import ABC, abstractmethod

from shared.plan.hop_deployment_id import HopDeploymentId
from worker.domain.hop.hop import DestinationHop
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ContributionSender(ABC):
    @abstractmethod
    async def send(
        self,
        contribution: HopInvocationContribution,
        destination_deployment_id: HopDeploymentId[DestinationHop],
    ) -> None: ...
