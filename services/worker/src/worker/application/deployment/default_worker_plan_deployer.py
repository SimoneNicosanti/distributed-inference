from typing import override

from artifacts.storage.artifact_store import ArtifactStore
from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId
from worker.application.hop_handler.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.outbound.catalog.partition_deployment.partition_deployment_registrar import (
    PartitionDeploymentRegistrar,
)
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
        partition_deployment_registrar: PartitionDeploymentRegistrar,
    ) -> None:
        self._worker_id = worker_id
        self._worker_plan_store = worker_plan_store
        self._artifact_store = artifact_store
        self._model_partition_metadata_reader = model_partition_metadata_reader
        self._partition_executor_factory = partition_executor_factory
        self._partition_executor_registry = partition_executor_registry
        self._partition_deployment_registrar = partition_deployment_registrar

    @override
    async def prepare_plan(self, plan: WorkerPlan) -> None:
        await self._create_partition_executors(plan)
        await self._worker_plan_store.put_plan(plan)

    async def _create_partition_executors(self, worker_plan: WorkerPlan) -> None:
        deployment_plan = worker_plan.deployment_plan

        for replica_id, resource_allocation in zip(
            deployment_plan.replica_ids,
            deployment_plan.allocations,
            strict=True,
        ):
            deployment_id = PartitionDeploymentId(
                replica_id=replica_id,
                worker_id=self._worker_id,
            )

            executor_exists = (
                await self._partition_executor_registry.check_partition_executor_exists(
                    replica_id
                )
            )
            if not executor_exists:
                artifact_ref = await self._model_partition_metadata_reader.get_artifact_ref_by_partition_id(
                    replica_id.partition_id
                )
                async with self._artifact_store.download_artifact(
                    artifact_ref
                ) as bundle:
                    partition_executor = await self._partition_executor_factory.create(
                        bundle,
                        resource_allocation,
                    )
                    await self._partition_executor_registry.register_partition_executor(
                        replica_id,
                        partition_executor,
                    )

            await self._partition_deployment_registrar.register_partition_deployment(
                plan_version=worker_plan.plan_version,
                deployment_id=deployment_id,
            )

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        await self._worker_plan_store.activate_plan(plan_version)
