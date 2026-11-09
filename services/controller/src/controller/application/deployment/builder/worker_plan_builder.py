from controller.application.deployment.builder.deployment_plan_builder import (
    DeploymentPlanBuilder,
)
from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.application.deployment.builder.failure_plan_builder import (
    FailurePlanBuilder,
)
from controller.application.deployment.builder.hop_routing_plan_builder import (
    HopRoutingPlanBuilder,
)
from controller.application.deployment.builder.partition_execution_plan_builder import (
    PartitionExecutionPlanBuilder,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.application.deployment.builder.scheduling_plan_builder import (
    SchedulingPlanBuilder,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.plan.worker_plan import WorkerPlan
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class WorkerPlanBuilder:
    ## TODO: Double check whole plan build mechanism
    @staticmethod
    def build_all(
        plan_version: PlanVersion,
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, WorkerPlan]:
        worker_ids = WorkerPlanBuilder._get_involved_workers(solution)
        deployment_plans = DeploymentPlanBuilder.build_all(
            worker_ids, solution, replica_ids_by_flow, endpoint_ids_by_flow
        )
        scheduling_plans = SchedulingPlanBuilder.build_all(
            worker_ids, solution, replica_ids_by_flow
        )
        execution_plans = PartitionExecutionPlanBuilder.build_all(
            worker_ids, solution, replica_ids_by_flow
        )
        routing_plans = HopRoutingPlanBuilder.build_all(
            worker_ids, solution, replica_ids_by_flow, endpoint_ids_by_flow
        )
        failure_plans = FailurePlanBuilder.build_all(
            worker_ids, solution, endpoint_ids_by_flow
        )

        return {
            worker_id: WorkerPlan(
                plan_version=plan_version,
                deployment_plan=deployment_plans[worker_id],
                scheduling_plan=scheduling_plans[worker_id],
                partition_execution_plan=execution_plans[worker_id],
                hop_routing_plan=routing_plans[worker_id],
                failure_plan=failure_plans[worker_id],
            )
            for worker_id in worker_ids
        }

    @staticmethod
    def _get_involved_workers(solution: OptimizationSolution) -> tuple[WorkerId, ...]:
        involved_workers: dict[WorkerId, None] = {}
        for flow in solution.flow_solutions:
            involved_workers.setdefault(flow.ingress_worker_id, None)
            involved_workers.setdefault(flow.sink_worker_id, None)
            for component in flow.executable_components:
                involved_workers.setdefault(component.worker_id, None)
        return tuple(involved_workers)
