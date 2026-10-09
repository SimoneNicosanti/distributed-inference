import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from shared.plan.partition_replica_id import PartitionReplicaId
from worker.application.partition_execution.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)


class DefaultPartitionExecutorRegistry(PartitionExecutorRegistry):
    @dataclass(frozen=True)
    class _PartitionExecutorBundle:
        partition_executor: PartitionExecutor
        lock: asyncio.Lock

    def __init__(self) -> None:
        self._partition_executors: dict[
            PartitionReplicaId,
            DefaultPartitionExecutorRegistry._PartitionExecutorBundle,
        ] = {}

    @override
    async def register_partition_executor(
        self,
        replica_id: PartitionReplicaId,
        partition_executor: PartitionExecutor,
    ) -> None:

        if replica_id in self._partition_executors:
            raise ValueError(
                f"Partition executor already present for replica {replica_id}"
            )

        bundle = DefaultPartitionExecutorRegistry._PartitionExecutorBundle(
            partition_executor, asyncio.Lock()
        )
        self._partition_executors[replica_id] = bundle

    @override
    async def unregister_partition_executor(
        self,
        replica_id: PartitionReplicaId,
    ) -> PartitionExecutor:
        if replica_id not in self._partition_executors:
            raise KeyError(f"Partition executor not found for replica {replica_id}")
        bundle = self._partition_executors.pop(replica_id)
        async with bundle.lock:
            return bundle.partition_executor

    @override
    @asynccontextmanager
    async def acquire_partition_executor(
        self, replica_id: PartitionReplicaId
    ) -> AsyncGenerator[PartitionExecutor]:
        if replica_id not in self._partition_executors:
            raise KeyError(f"Partition executor not found for replica {replica_id}")
        bundle = self._partition_executors[replica_id]

        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.partition_executor

    @override
    async def check_partition_executor_exists(
        self, replica_id: PartitionReplicaId
    ) -> bool:
        return replica_id in self._partition_executors
