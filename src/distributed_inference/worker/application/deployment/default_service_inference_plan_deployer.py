from typing import override

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.ports.inbound.service_inference_plan_deployer import (
    ServiceInferencePlanDeployer,
)


class DefaultServiceInferencePlanDeployer(ServiceInferencePlanDeployer):
    def __init__(self, artifact_materializer: ArtifactMaterializer) -> None:

        self._artifact_materializer = artifact_materializer
        pass

    @override
    async def deploy_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None:

        ## 1. Publish service inference plan to the service inference plan store
        ## 2. Let everyone update based on the new plan
        ## 3. For each sub model in the plan, materialize the artifact
        ## 4. Call the executor factory
        ## 5. Register the sub model in the executor registry

        pass
