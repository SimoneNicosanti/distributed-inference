import asyncio
from asyncio import Future
from typing import Any, override

from distributed_inference.application.scheduling.contracts.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.application.worker.contracts.deployment.inference_plan_preparer import (
    InferencePlanPreparer,
)
from distributed_inference.application.worker.contracts.deployment.inference_plan_store import (
    InferencePlanStore,
)
from distributed_inference.application.worker.contracts.execution.inference.inference_request_scheduler import (
    InferenceRequestScheduler,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
)
from distributed_inference.domain.plan import InferencePlanVersion, ServiceInferencePlan


class PlanPreparerInferenceRequestScheduler(
    InferenceRequestScheduler, InferencePlanPreparer
):
    def __init__(self, inference_plan_store: InferencePlanStore) -> None:
        self._condition = asyncio.Condition()

        ## We use the inference_plan_store as a sync point
        self._inference_plan_store = inference_plan_store
        self._inference_request_schedulers: dict[
            InferencePlanVersion, RequestScheduler
        ] = {}

    @override
    async def enqueue(self, request: InferenceRequest, future: Future[Any]) -> None:
        plan_version = request.inference_plan_version
        async with self._condition:
            if self._inference_request_schedulers.get(plan_version, None) is not None:
                await self._inference_request_schedulers[plan_version].enqueue(
                    request, future
                )
                self._condition.notify(n=1)
            else:
                raise ValueError(f"Plan version {plan_version} does not exist")
        return None

    @override
    async def dequeue(self) -> tuple[InferenceRequest, Future[Any]]:
        async with self._condition:
            while True:
                sorted_versions = sorted(self._inference_request_schedulers)

                lengths = await asyncio.gather(
                    *(
                        self._inference_request_schedulers[version].length()
                        for version in sorted_versions
                    )
                )

                for version, length in zip(
                    sorted_versions,
                    lengths,
                    strict=True,
                ):
                    if length > 0:
                        scheduler = self._inference_request_schedulers[version]
                        ## If there is a non zero length, we dequeue the one at max priority
                        return await scheduler.dequeue()

                ## Otherwise, we wait (releasing the lock)
                ## In this case, this might not be necessary, because we have only one consumer, but better to double check
                await self._condition.wait()

    @override
    async def length(self) -> int:
        async with self._condition:
            total_length = 0

            for scheduler in self._inference_request_schedulers.values():
                total_length += await scheduler.length()

            return total_length

    @override
    async def prepare_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## We make this implement this interface so it can switch plan when needed
        raise NotImplementedError
        ## TODO: Implement this method
        ## 1. If Real scheduler exists, then switch plan
        ## 2. If Real scheduler does not exist, then create it and give the plan or an abstraction of it

        ## Possible policy to switch:
        ## 1. The older plan needs to have an higher priority: we need to complete the requests for that plan in order to free resources
        ## 2. The newer plan will have a lower priority and it will be executed once all requests of the older plan have been completed

        ## We don't need to pass the whole plan to the scheduler, just the mapping (FlowId, SubModelId) -> Priority
