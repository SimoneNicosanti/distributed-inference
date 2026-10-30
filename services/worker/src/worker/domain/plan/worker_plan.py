from dataclasses import dataclass

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.deployment_plan import DeploymentPlan
from worker.domain.plan.hop_routing_plan import HopRoutingPlan
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.scheduling_plan import SchedulingPlan


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    plan_version: PlanVersion
    deployment_plan: DeploymentPlan
    scheduling_plan: SchedulingPlan
    partition_execution_plan: PartitionExecutionPlan
    hop_routing_plan: HopRoutingPlan

    def __post_init__(self) -> None:
        deployed_replica_ids = set(self.deployment_plan.replica_ids)

        deployed_partition_ids = {
            replica_id.partition_id for replica_id in deployed_replica_ids
        }
        if deployed_partition_ids != set(self.partition_execution_plan.partition_ids):
            raise ValueError("Each deployed partition must have an execution scheme")
