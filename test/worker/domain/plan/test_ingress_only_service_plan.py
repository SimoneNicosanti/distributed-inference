from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId
from worker.domain.plan.ingress.flow_plan import FlowPlan, IngressInputRoute
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.service_plan import ServicePlan
from worker.domain.plan.worker.deployment_plan import DeploymentPlan
from worker.domain.plan.worker.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.worker.routing_plan import RoutingPlan
from worker.domain.plan.worker.scheduling_plan import SchedulingPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


def test_ingress_only_service_plan_accepts_empty_worker_plan() -> None:
    user_id = UserId()
    flow_id = FlowId(user_id=user_id)
    model_variant_id = ModelVariantId(
        model_id=ModelId(owner_id=user_id, model_name="model"),
        variant_tag="variant",
    )
    target_replica_id = PartitionReplicaId(
        partition_id=ModelPartitionId(
            model_variant_id=model_variant_id,
            layers=("compute",),
        ),
        replica_idx=0,
    )
    plan_version = PlanVersion(version_number=1)

    plan = ServicePlan(
        worker_plan=WorkerPlan(
            plan_version=plan_version,
            deployment_plan=DeploymentPlan(replica_ids=(), allocations=()),
            scheduling_plan=SchedulingPlan(priority_keys=(), priorities=()),
            partition_execution_plan=PartitionExecutionPlan(
                partition_ids=(),
                schemes=(),
            ),
            routing_plan=RoutingPlan(replica_ids=(), routings=()),
        ),
        ingress_plan=IngressPlan(
            plan_version=plan_version,
            flow_ids=(flow_id,),
            flow_plans=(
                FlowPlan(
                    model_variant_id=model_variant_id,
                    input_routes=(
                        IngressInputRoute(
                            target_replica_id=target_replica_id,
                            tensors=("input",),
                        ),
                    ),
                    result_tensors=("result",),
                ),
            ),
        ),
    )

    assert plan.worker_plan.deployment_plan.replica_ids == ()
    assert plan.ingress_plan is not None
