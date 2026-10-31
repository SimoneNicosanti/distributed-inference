import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from worker.application.hop_handler.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)
from worker.domain.hop.hop import ExecutableHop


class DefaultPartitionExecutorRegistry(PartitionExecutorRegistry):
    @dataclass(frozen=True)
    class _PartitionExecutorBundle:
        partition_executor: PartitionExecutor
        lock: asyncio.Lock

    def __init__(self) -> None:
        self._partition_executors: dict[
            ExecutableHop,
            DefaultPartitionExecutorRegistry._PartitionExecutorBundle,
        ] = {}

    @override
    async def register_partition_executor(
        self,
        executable_hop: ExecutableHop,
        partition_executor: PartitionExecutor,
    ) -> None:
        if executable_hop in self._partition_executors:
            raise ValueError(
                f"Partition executor already present for hop {executable_hop}"
            )

        bundle = DefaultPartitionExecutorRegistry._PartitionExecutorBundle(
            partition_executor, asyncio.Lock()
        )
        self._partition_executors[executable_hop] = bundle

    @override
    async def unregister_partition_executor(
        self,
        executable_hop: ExecutableHop,
    ) -> PartitionExecutor:
        if executable_hop not in self._partition_executors:
            raise KeyError(f"Partition executor not found for hop {executable_hop}")
        bundle = self._partition_executors.pop(executable_hop)
        async with bundle.lock:
            return bundle.partition_executor

    @override
    @asynccontextmanager
    async def acquire_partition_executor(
        self, executable_hop: ExecutableHop
    ) -> AsyncGenerator[PartitionExecutor]:
        if executable_hop not in self._partition_executors:
            raise KeyError(f"Partition executor not found for hop {executable_hop}")
        bundle = self._partition_executors[executable_hop]

        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.partition_executor

    @override
    async def check_partition_executor_exists(
        self, executable_hop: ExecutableHop
    ) -> bool:
        return executable_hop in self._partition_executors
