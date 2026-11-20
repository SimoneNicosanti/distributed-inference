from typing import override

from artifacts.storage.artifact_store import ArtifactDownloadError, ArtifactStore
from shared.plan.hop_logical_id import PlannedPartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId
from worker.application.hop_handler.abc.hop_executor_initializer import (
    HopExecutorInitError,
    HopExecutorInitializer,
)
from worker.application.ports.outbound.hop_execution.hop_executor import HopExecutor
from worker.application.ports.outbound.hop_execution.hop_executor_factory import (
    HopExecutorFactory,
    PartitionExecutorCreationError,
)
from worker.application.ports.outbound.model_metadata.model_partition_metadata_reader import (
    ArtifactReferenceReadError,
    ModelPartitionMetadataReader,
)
from worker.application.ports.outbound.plan_reader.worker_plan_reader import (
    WorkerPlanReadError,
)
from worker.application.ports.outbound.plan_reader.worker_plan_view_readers import (
    DeploymentPlanReader,
)
from worker.domain.plan.deployment_plan import NoResourceAllocationForHopError


class DefaultPartitionExecutorInitializer(HopExecutorInitializer):
    def __init__(
        self,
        worker_id: WorkerId,
        deployment_plan_reader: DeploymentPlanReader,
        model_partition_metadata_reader: ModelPartitionMetadataReader,
        artifact_store: ArtifactStore,
        partition_executor_factory: HopExecutorFactory,
    ) -> None:
        self._worker_id = worker_id
        self._deployment_plan_reader = deployment_plan_reader
        self._model_partition_metadata_reader = model_partition_metadata_reader
        self._artifact_store = artifact_store
        self._partition_executor_factory = partition_executor_factory

    @override
    async def initialize(
        self, plan_version: PlanVersion, hop_id: PlannedPartitionReplicaId
    ) -> HopExecutor:

        try:
            deployment_plan = (
                await self._deployment_plan_reader.get_deployment_plan_by_version(
                    plan_version
                )
            )

            resource_allocation = deployment_plan.get_allocation_by_executable_hop(
                hop_id
            )

            artifact_ref = await (
                self._model_partition_metadata_reader.get_artifact_ref_by_partition_id(
                    hop_id.partition_id
                )
            )

            async with self._artifact_store.download_artifact(artifact_ref) as bundle:
                ## TODO: Refactor this for future devolpment: this is not generic at all
                partition_executor = await self._partition_executor_factory.create(
                    bundle,
                    resource_allocation,
                )
                return partition_executor
        except (
            WorkerPlanReadError,
            NoResourceAllocationForHopError,
            ArtifactReferenceReadError,
            ArtifactDownloadError,
            PartitionExecutorCreationError,
        ) as e:
            raise HopExecutorInitError("Could not init hop executor") from e
