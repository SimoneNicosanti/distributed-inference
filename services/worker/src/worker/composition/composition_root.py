from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from integration.service_instance.worker_instance import WorkerInstance
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.inbound.ingress.model_invocation_coordinator import (
    ModelInvocationCoordinator,
)
from worker.application.ports.inbound.partition_invocation.partition_invocation_coordinator import (
    PartitionInvocationCoordinator,
)
from worker.application.ports.inbound.sink.model_pass_result_contribution_sink import (
    ModelPassResultContributionSink,
)
from worker.application.worker_identity.abc.worker_identity_query import (
    WorkerIdentityQuery,
)
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.sink.sink_plan import SinkPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


@dataclass(frozen=True)
class WorkerComposition:
    worker_plan_deployer: PlanDeployer[WorkerPlan]
    partition_invocation_coordinator: PartitionInvocationCoordinator
    worker_identity_query: WorkerIdentityQuery
    ingress_plan_deployer: PlanDeployer[IngressPlan] | None = None
    sink_plan_deployer: PlanDeployer[SinkPlan] | None = None
    model_invocation_coordinator: ModelInvocationCoordinator | None = None
    result_contribution_sink: ModelPassResultContributionSink | None = None


class CompositionRoot(ABC):
    @abstractmethod
    @asynccontextmanager
    def compose(
        self,
        service_instance: WorkerInstance,
    ) -> AsyncGenerator[WorkerComposition]: ...
