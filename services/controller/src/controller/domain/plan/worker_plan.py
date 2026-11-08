from dataclasses import dataclass

from controller.domain.plan.deployment_plan import DeploymentPlan
from controller.domain.plan.failure_plan import FailurePlan
from controller.domain.plan.hop_routing_plan import HopRoutingPlan
from controller.domain.plan.partition_execution_plan import PartitionExecutionPlan
from controller.domain.plan.scheduling_plan import SchedulingPlan
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    plan_version: PlanVersion
    deployment_plan: DeploymentPlan
    scheduling_plan: SchedulingPlan
    partition_execution_plan: PartitionExecutionPlan
    hop_routing_plan: HopRoutingPlan
    failure_plan: FailurePlan

    def __post_init__(self) -> None:
        deployed_replicas = set(self.deployment_plan.executable_hops)
        routed_hops = {address.hop for address in self.hop_routing_plan.hop_addresses}
        if set(self.deployment_plan.deployed_hops) != routed_hops:
            raise ValueError("Deployment and routing plans must contain the same hops")

        routed_flows = {
            address.flow_id for address in self.hop_routing_plan.hop_addresses
        }
        if set(self.failure_plan.flow_ids) != routed_flows:
            raise ValueError(
                "Failure plan must cover exactly the worker's routed flows"
            )

        deployed_partitions = {replica.partition_id for replica in deployed_replicas}
        if deployed_partitions != set(self.partition_execution_plan.partition_ids):
            raise ValueError("Each deployed partition must have an execution scheme")

        scheduled_replicas = {
            key.executable_hop for key in self.scheduling_plan.priority_keys
        }
        if not scheduled_replicas.issubset(deployed_replicas):
            raise ValueError("Scheduling plan references undeployed replicas")
