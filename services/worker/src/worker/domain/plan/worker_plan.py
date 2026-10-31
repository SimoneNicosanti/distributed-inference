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
        deployed_executable_hops = set(self.deployment_plan.executable_hops)

        deployed_partition_ids = {
            executable_hop.partition_id for executable_hop in deployed_executable_hops
        }
        if deployed_partition_ids != set(self.partition_execution_plan.partition_ids):
            raise ValueError("Each deployed partition must have an execution scheme")
