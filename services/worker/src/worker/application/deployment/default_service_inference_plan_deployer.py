from typing import override

from artifacts.storage.artifact_store import ArtifactStore
from shared.plan.plan import ServiceInferencePlan
from worker.application.deployment.abc.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.partition_execution.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.inbound.deployment.service_inference_plan_applier import (
    ServiceInferencePlanApplier,
)
from worker.application.ports.outbound.partition_execution.partition_executor_factory import (
    PartitionExecutorFactory,
)
from worker.application.ports.outbound.plan_store.service_inference_plan_store import (
    ServiceInferencePlanStore,
)


class DefaultServiceInferencePlanDeployer(
    ServiceInferencePlanApplier, ServiceInferencePlanPreparer
):
    def __init__(
        self,
        service_inference_plan_store: ServiceInferencePlanStore,
        artifact_store: ArtifactStore,
        partition_executor_factory: PartitionExecutorFactory,
        partition_executor_registry: PartitionExecutorRegistry,
        service_inference_plan_preparers: list[ServiceInferencePlanPreparer],
    ):
        self._service_inference_plan_store = service_inference_plan_store
        self._artifact_store = artifact_store
        self._partition_executor_factory = partition_executor_factory
        self._partition_executor_registry = partition_executor_registry
        self._service_inference_plan_preparers = service_inference_plan_preparers

    ## This is the first call done by the control plane
    ## Every worker prepares the new plan
    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## TODO: In order for this to work correctly, they should be called sequentially
        ## Otherwise we might have a race condition

        ## Create partition executors required by this plan.
        await self._create_partition_executors(service_inference_plan)

        ## We publish the new plan to the store
        await self._service_inference_plan_store.put_service_inference_plan(
            service_inference_plan
        )
        ## We tell everyone to prepare for the new plan
        for preparer in self._service_inference_plan_preparers:
            await preparer.prepare_service_inference_plan(service_inference_plan)

    async def _create_partition_executors(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        for partition_deployment in service_inference_plan.sub_model_deployments:
            resource_allocation = partition_deployment.resource_allocation

            ## If we already have the deployment, we skip the rebuild
            ## Same partition, same worker, same resources.
            ## TODO: We might need to enforce a stronger policy to avoid duplication
            ## Especially in case of not sequential calls to the prepare API (lock in the deployer)
            if await self._partition_executor_registry.check_partition_executor_exists(
                partition_deployment
            ):
                continue

            artifact_ref = partition_deployment.artifact_ref
            async with self._artifact_store.download_artifact(artifact_ref) as bundle:
                partition_executor = await self._partition_executor_factory.create(
                    bundle, resource_allocation
                )
                await self._partition_executor_registry.register_partition_executor(
                    partition_deployment, partition_executor
                )

    ## This is the second message sent by the control plane
    ## It is used to commit the new plan
    @override
    async def apply_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None:
        ## We activate the new plan in the store
        ## Everyone referencing the active plan will use that version
        await self._service_inference_plan_store.activate_service_inference_plan(
            service_inference_plan.plan_version
        )

        ## TODO: We should handle eviction of old models or unused deployments
        ## What if we do not have enough resources to run the new plan?
        ## We should handle this in some way, maybe using a three steps process
        ## 1. Consume requests belonging to the old plan
        ## 2. Check changed deployments
        ## 3. Evict removed deployments
        ## 4. Create new deployments
        ## This is ok assuming that the control plane handles correctly the resources
        ## Main problem here is memory

        ## Regarding possible state of LLMs:
        ## It is related to the attention layers, so we can just check where those layers have been assigned
        ## Or if the model has changed we can recompute the KV-Cache from scratch using already generated tokens
