from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from integration.service_instance.worker_instance import WorkerInstance
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.inbound.hop_contribution_rcv.hop_invocation_contribution_receive_coordinator import (
    HopInvocationContributionReceiveCoordinator,
)
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.application.worker_identity.abc.worker_identity_query import (
    WorkerIdentityQuery,
)
from worker.domain.plan.worker_plan import WorkerPlan


@dataclass(frozen=True)
class WorkerComposition:
    worker_plan_deployer: PlanDeployer[WorkerPlan]
    hop_contribution_receiver: HopInvocationContributionReceiveCoordinator
    model_pass_result_retriever: ModelPassResultRetriever
    worker_identity_query: WorkerIdentityQuery


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(
        self,
        service_instance: WorkerInstance,
    ) -> AsyncGenerator[WorkerComposition]: ...
