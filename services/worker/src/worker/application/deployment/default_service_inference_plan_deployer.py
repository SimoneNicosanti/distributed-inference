from typing import override

from shared.artifact.artifact_ref import build_artifact_ref
from artifacts.storage.artifact_store import ArtifactStore
from shared.plan.plan import ServiceInferencePlan
from worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.ports.inbound.service_inference_plan_applier import (
    ServiceInferencePlanApplier,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.ports.outbound.sub_model_executor_factory import (
    SubModelExecutorFactory,
)
from worker.application.sub_model_execution.contracts.sub_model_executor_registry import (
    SubModelExecutorRegistry,
)


class DefaultServiceInferencePlanDeployer(
    ServiceInferencePlanApplier, ServiceInferencePlanPreparer
):
    def __init__(
        self,
        service_inference_plan_store: ServiceInferencePlanStore,
        artifact_store: ArtifactStore,
        sub_model_executor_factory: SubModelExecutorFactory,
        sub_model_executor_registry: SubModelExecutorRegistry,
        service_inference_plan_preparers: list[ServiceInferencePlanPreparer],
    ):
        self._service_inference_plan_store = service_inference_plan_store
        self._artifact_store = artifact_store
        self._sub_model_executor_factory = sub_model_executor_factory
        self._sub_model_executor_registry = sub_model_executor_registry
        self._service_inference_plan_preparers = service_inference_plan_preparers

    ## This is the first call done by the control plane
    ## Every worker prepares the new plan
    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## TODO: In order for this to work correctly, they should be called sequentially
        ## Otherwise we might have a race condition

        ## We create the sub-model executors that are needed for this plan
        await self._create_sub_model_executors(service_inference_plan)

        ## We publish the new plan to the store
        await self._service_inference_plan_store.put_service_inference_plan(
            service_inference_plan
        )
        ## We tell everyone to prepare for the new plan
        for preparer in self._service_inference_plan_preparers:
            await preparer.prepare_service_inference_plan(service_inference_plan)

    async def _create_sub_model_executors(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        for deployment in service_inference_plan.sub_model_deployments:
            sub_model_id = deployment.sub_model_id
            resource_allocation = deployment.resource_allocation

            ## If we already have the deployment, we skip the rebuild
            ## Same sub-model, same worker, same resources
            ## TODO: We might need to enforce a stronger policy to avoid duplication
            ## Especially in case of not sequential calls to the prepare API (lock in the deployer)
            if await self._sub_model_executor_registry.check_sub_model_executor(
                deployment
            ):
                continue

            artifact_ref = await build_artifact_ref(sub_model_id.model_dump_json())
            async with self._artifact_store.download_artifact(artifact_ref) as bundle:
                sub_model_executor = (
                    await self._sub_model_executor_factory.create_sub_model_executor(
                        bundle, resource_allocation
                    )
                )
                await self._sub_model_executor_registry.register_sub_model_executor(
                    deployment, sub_model_executor
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
