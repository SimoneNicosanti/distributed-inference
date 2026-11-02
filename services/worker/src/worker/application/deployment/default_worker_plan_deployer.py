from typing import override

from artifacts.storage.artifact_store import ArtifactStore
from catalog.api.hop.hop_deployment_registrar import (
    HopDeploymentRegistrar,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId
from worker.application.hop_handler.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.outbound.model_metadata.model_partition_metadata_reader import (
    ModelPartitionMetadataReader,
)
from worker.application.ports.outbound.partition_execution.partition_executor_factory import (
    PartitionExecutorFactory,
)
from worker.application.ports.outbound.plan_store.plan_store_writer import (
    PlanStoreWriter,
)
from worker.domain.plan.worker_plan import WorkerPlan


class DefaultWorkerPlanDeployer(PlanDeployer[WorkerPlan]):
    def __init__(
        self,
        worker_id: WorkerId,
        worker_plan_store: PlanStoreWriter[WorkerPlan],
        artifact_store: ArtifactStore,
        model_partition_metadata_reader: ModelPartitionMetadataReader,
        partition_executor_factory: PartitionExecutorFactory,
        partition_executor_registry: PartitionExecutorRegistry,
        hop_deployment_registrar: HopDeploymentRegistrar,
    ) -> None:
        self._worker_id = worker_id
        self._worker_plan_store = worker_plan_store
        self._artifact_store = artifact_store
        self._model_partition_metadata_reader = model_partition_metadata_reader
        self._partition_executor_factory = partition_executor_factory
        self._partition_executor_registry = partition_executor_registry
        self._hop_deployment_registrar = hop_deployment_registrar

    @override
    async def prepare_plan(self, plan: WorkerPlan) -> None:
        await self._create_partition_executors(plan)
        await self._register_hop_deployments(plan)
        await self._worker_plan_store.put_plan(plan)

    async def _create_partition_executors(self, worker_plan: WorkerPlan) -> None:
        deployment_plan = worker_plan.deployment_plan

        for executable_hop, resource_allocation in zip(
            deployment_plan.executable_hops,
            deployment_plan.allocations,
            strict=True,
        ):
            executor_exists = (
                await self._partition_executor_registry.check_partition_executor_exists(
                    worker_plan.plan_version, executable_hop
                )
            )
            if not executor_exists:
                artifact_ref = await self._model_partition_metadata_reader.get_artifact_ref_by_partition_id(
                    executable_hop.partition_id
                )
                async with self._artifact_store.download_artifact(
                    artifact_ref
                ) as bundle:
                    partition_executor = await self._partition_executor_factory.create(
                        bundle,
                        resource_allocation,
                    )
                    await self._partition_executor_registry.register_partition_executor(
                        worker_plan.plan_version,
                        executable_hop,
                        partition_executor,
                    )

    async def _register_hop_deployments(self, worker_plan: WorkerPlan) -> None:
        planned_hops = worker_plan.deployment_plan.planned_hops
        await self._hop_deployment_registrar.register_worker_for_planned_invocable_hop(
            plan_version=worker_plan.plan_version,
            planned_hops=planned_hops,
            worker_id=self._worker_id,
        )

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        await self._worker_plan_store.activate_plan(plan_version)
