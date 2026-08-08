import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from distributed_inference.domain.plan import SubModelDeployment
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)
from distributed_inference.worker.application.sub_model_execution.contracts.sub_model_executor_registry import (
    SubModelExecutorRegistry,
)


class DefaultSubModelExecutorRegistry(SubModelExecutorRegistry):
    @dataclass(frozen=True)
    class _SubModelExecutorBundle:
        sub_model_executor: SubModelExecutor
        lock: asyncio.Lock

    def __init__(self) -> None:
        self._sub_model_executors: dict[
            SubModelDeployment, DefaultSubModelExecutorRegistry._SubModelExecutorBundle
        ] = {}

    @override
    async def register_sub_model_executor(
        self,
        sub_model_deployment: SubModelDeployment,
        sub_model_executor: SubModelExecutor,
    ) -> None:

        if sub_model_deployment in self._sub_model_executors:
            raise ValueError(
                f"Sub-model executor already present for sub-model deployment {sub_model_deployment}"
            )

        bundle = DefaultSubModelExecutorRegistry._SubModelExecutorBundle(
            sub_model_executor, asyncio.Lock()
        )
        self._sub_model_executors[sub_model_deployment] = bundle

    @override
    async def unregister_sub_model_executor(
        self,
        sub_model_deployment: SubModelDeployment,
    ) -> SubModelExecutor:
        if sub_model_deployment not in self._sub_model_executors:
            raise KeyError(
                f"Sub-model executor not found for sub-model deployment {sub_model_deployment}"
            )
        bundle = self._sub_model_executors.pop(sub_model_deployment)
        async with bundle.lock:
            return bundle.sub_model_executor

    @override
    @asynccontextmanager
    async def acquire_sub_model_executor(
        self, sub_model_deployment: SubModelDeployment
    ) -> AsyncGenerator[SubModelExecutor]:
        if sub_model_deployment not in self._sub_model_executors:
            raise KeyError(
                f"Sub-model executor not found for sub-model deployment {sub_model_deployment}"
            )
        bundle = self._sub_model_executors[sub_model_deployment]

        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.sub_model_executor

    @override
    async def check_sub_model_executor(
        self, sub_model_deployment: SubModelDeployment
    ) -> bool:
        return sub_model_deployment in self._sub_model_executors
