from typing import override

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_store.domain.artifact_key import SubModelArtifactKey
from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.model_manager.domain.sub_model import SubModelDeploymentId
from distributed_inference.worker.application.ports.inbound.service_inference_plan_deployer import (
    ServiceInferencePlanDeployer,
)
from distributed_inference.worker.application.ports.outbound.sub_model_executor_factory import (
    SubModelExecutorFactory,
)
from distributed_inference.worker.application.sub_model_execution.contracts.sub_model_executor_registry import (
    SubModelExecutorRegistry,
)


class DefaultServiceInferencePlanDeployer(ServiceInferencePlanDeployer):
    def __init__(
        self,
        artifact_materializer: ArtifactMaterializer,
        sub_model_executor_factory: SubModelExecutorFactory,
        sub_model_executor_registry: SubModelExecutorRegistry,
    ):

        self._artifact_materializer = artifact_materializer
        self._sub_model_executor_factory = sub_model_executor_factory
        self._sub_model_executor_registry = sub_model_executor_registry

    @override
    async def deploy_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None:

        ## We materialize (downloading if needed) the sub-models
        for replica_id in service_inference_plan.sub_model_replicas:
            sub_model_id = replica_id.sub_model_id
            deployment_options = service_inference_plan.deployment_options[replica_id]

            artifact_key = SubModelArtifactKey(id=sub_model_id)
            async with self._artifact_materializer.materialize_artifact(
                artifact_key
            ) as materialized_artifact:
                sub_model_executor = (
                    await self._sub_model_executor_factory.create_sub_model_executor(
                        materialized_artifact, deployment_options
                    )
                )
                deployment_id = SubModelDeploymentId(
                    sub_model_replica_id=replica_id,
                    service_id=service_inference_plan.service_id,
                )
                await self._sub_model_executor_registry.register_sub_model_executor(
                    deployment_id, sub_model_executor
                )

        ## 1. Publish service inference plan to the service inference plan store
        ## 2. Let everyone update based on the new plan
