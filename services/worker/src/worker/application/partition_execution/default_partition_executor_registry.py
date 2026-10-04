import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from shared.plan.plan import PartitionDeployment
from worker.application.partition_execution.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.partition_executor import (
    PartitionExecutor,
)


class DefaultPartitionExecutorRegistry(PartitionExecutorRegistry):
    @dataclass(frozen=True)
    class _PartitionExecutorBundle:
        partition_executor: PartitionExecutor
        lock: asyncio.Lock

    def __init__(self) -> None:
        self._partition_executors: dict[
            PartitionDeployment,
            DefaultPartitionExecutorRegistry._PartitionExecutorBundle,
        ] = {}

    @override
    async def register_partition_executor(
        self,
        partition_deployment: PartitionDeployment,
        partition_executor: PartitionExecutor,
    ) -> None:

        if partition_deployment in self._partition_executors:
            raise ValueError(
                f"Partition executor already present for deployment {partition_deployment}"
            )

        bundle = DefaultPartitionExecutorRegistry._PartitionExecutorBundle(
            partition_executor, asyncio.Lock()
        )
        self._partition_executors[partition_deployment] = bundle

    @override
    async def unregister_partition_executor(
        self,
        partition_deployment: PartitionDeployment,
    ) -> PartitionExecutor:
        if partition_deployment not in self._partition_executors:
            raise KeyError(
                f"Partition executor not found for deployment {partition_deployment}"
            )
        bundle = self._partition_executors.pop(partition_deployment)
        async with bundle.lock:
            return bundle.partition_executor

    @override
    @asynccontextmanager
    async def acquire_partition_executor(
        self, partition_deployment: PartitionDeployment
    ) -> AsyncGenerator[PartitionExecutor]:
        if partition_deployment not in self._partition_executors:
            raise KeyError(
                f"Partition executor not found for deployment {partition_deployment}"
            )
        bundle = self._partition_executors[partition_deployment]

        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.partition_executor

    @override
    async def check_partition_executor_exists(
        self, partition_deployment: PartitionDeployment
    ) -> bool:
        return partition_deployment in self._partition_executors
