from dataclasses import dataclass

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.worker.deployment_plan import DeploymentPlan
from worker.domain.plan.worker.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.worker.routing_plan import RoutingPlan
from worker.domain.plan.worker.scheduling_plan import SchedulingPlan


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    plan_version: PlanVersion
    deployment_plan: DeploymentPlan
    scheduling_plan: SchedulingPlan
    partition_execution_plan: PartitionExecutionPlan
    routing_plan: RoutingPlan

    def __post_init__(self) -> None:
        deployed_replica_ids = set(self.deployment_plan.replica_ids)

        if deployed_replica_ids != set(self.routing_plan.replica_ids):
            raise ValueError(
                "Deployment and routing plans must contain the same replicas"
            )

        deployed_partition_ids = {
            replica_id.partition_id for replica_id in deployed_replica_ids
        }
        if deployed_partition_ids != set(self.partition_execution_plan.partition_ids):
            raise ValueError("Each deployed partition must have an execution scheme")

        scheduled_replica_ids = {
            key.partition_replica_id for key in self.scheduling_plan.priority_keys
        }
        if not scheduled_replica_ids.issubset(deployed_replica_ids):
            raise ValueError("Scheduling plan references undeployed replicas")
