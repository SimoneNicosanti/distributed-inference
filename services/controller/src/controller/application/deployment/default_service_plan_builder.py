from typing import override

from controller.application.deployment.abc.service_plan_builder import (
    ServicePlanBuilder,
)
from controller.application.deployment.builder.default_ingress_plan_builder import (
    DefaultIngressPlanBuilder,
)
from controller.application.deployment.builder.default_worker_plan_builder import (
    DefaultWorkerPlanBuilder,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    PartitionReplicaIdBuilder,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.plan.service_plan import ServicePlan
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class DefaultServicePlanBuilder(ServicePlanBuilder):
    def __init__(self) -> None:
        self._replica_id_builder = PartitionReplicaIdBuilder()
        self._worker_plan_builder = DefaultWorkerPlanBuilder()
        self._ingress_plan_builder = DefaultIngressPlanBuilder()

    @override
    async def build_all_plans_from_solution(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
    ) -> dict[WorkerId, ServicePlan]:
        replica_ids_by_flow = (
            self._replica_id_builder.build_all_partition_replica_ids_from_solution(
                solution
            )
        )
        worker_plans = self._worker_plan_builder.build_all(
            plan_version,
            solution,
            replica_ids_by_flow,
        )
        ingress_plans = self._ingress_plan_builder.build_all(
            plan_version,
            solution,
            replica_ids_by_flow,
        )

        return {
            worker_id: ServicePlan(
                worker_plan=worker_plan,
                ingress_plan=ingress_plans.get(worker_id),
            )
            for worker_id, worker_plan in worker_plans.items()
        }
