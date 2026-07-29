from typing import override

import aiorwlock

from distributed_inference.domain.plan import InferencePlanVersion, ServiceInferencePlan
from distributed_inference.worker.application.ports.outbound.inference_plan_store import (
    InferencePlanStore,
)


class InMemoryInferencePlanStore(InferencePlanStore):
    def __init__(self) -> None:
        self._lock = aiorwlock.RWLock()
        self._latest_inference_plan: ServiceInferencePlan | None = None
        self._active_inference_plan: ServiceInferencePlan | None = None
        self._inference_plan_by_version: dict[
            InferencePlanVersion, ServiceInferencePlan
        ] = {}

    @override
    async def put_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:

        async with self._lock.writer_lock:
            if service_inference_plan.plan_version in self._inference_plan_by_version:
                raise ValueError("Inference plan already exists")

            if self._latest_inference_plan is None:
                self._latest_inference_plan = service_inference_plan
            else:
                if (
                    service_inference_plan.plan_version
                    > self._latest_inference_plan.plan_version
                ):
                    self._latest_inference_plan = service_inference_plan

            self._inference_plan_by_version[service_inference_plan.plan_version] = (
                service_inference_plan
            )

    @override
    async def get_inference_plan_by_version(
        self, version: InferencePlanVersion
    ) -> ServiceInferencePlan | None:
        async with self._lock.reader_lock:
            return self._inference_plan_by_version.get(version, None)

    @override
    async def get_latest_inference_plan(self) -> ServiceInferencePlan | None:
        async with self._lock.reader_lock:
            return self._latest_inference_plan

    @override
    async def get_active_inference_plan(self) -> ServiceInferencePlan | None:
        async with self._lock.reader_lock:
            return self._active_inference_plan

    @override
    async def activate_inference_plan(self, version: InferencePlanVersion) -> None:
        async with self._lock.writer_lock:
            if version not in self._inference_plan_by_version:
                raise ValueError("Inference plan does not exist")

            self._active_inference_plan = self._inference_plan_by_version[version]
