from dataclasses import dataclass

from controller.domain.plan.deployment_plan import DeploymentPlan
from controller.domain.plan.partition_execution_plan import PartitionExecutionPlan
from controller.domain.plan.routing_plan import RoutingPlan
from controller.domain.plan.scheduling_plan import SchedulingPlan
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    plan_version: PlanVersion
    deployment_plan: DeploymentPlan
    scheduling_plan: SchedulingPlan
    partition_execution_plan: PartitionExecutionPlan
    routing_plan: RoutingPlan

    def __post_init__(self) -> None:
        deployed_replicas = set(self.deployment_plan.replica_ids)
        if deployed_replicas != set(self.routing_plan.replica_ids):
            raise ValueError(
                "Deployment and routing plans must contain the same replicas"
            )

        deployed_partitions = {replica.partition_id for replica in deployed_replicas}
        if deployed_partitions != set(self.partition_execution_plan.partition_ids):
            raise ValueError("Each deployed partition must have an execution scheme")

        scheduled_replicas = {
            key.partition_replica_id for key in self.scheduling_plan.priority_keys
        }
        if not scheduled_replicas.issubset(deployed_replicas):
            raise ValueError("Scheduling plan references undeployed replicas")
