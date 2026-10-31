from shared.plan.plan_version import PlanVersion
from worker.domain.plan.deployment_plan import DeploymentPlan
from worker.domain.plan.hop_routing_plan import HopRoutingPlan
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.scheduling_plan import SchedulingPlan
from worker.domain.plan.worker_plan import WorkerPlan


def test_worker_plan_accepts_empty_subplans() -> None:
    plan = WorkerPlan(
        plan_version=PlanVersion(version_number=1),
        deployment_plan=DeploymentPlan(
            executable_hops=(),
            allocations=(),
            ingress_hops=(),
            sink_hops=(),
        ),
        scheduling_plan=SchedulingPlan(priority_keys=(), priorities=()),
        partition_execution_plan=PartitionExecutionPlan(
            partition_ids=(),
            schemes=(),
        ),
        hop_routing_plan=HopRoutingPlan(hop_addresses=(), routings=()),
    )

    assert plan.deployment_plan.executable_hops == ()
    assert plan.hop_routing_plan.hop_addresses == ()
