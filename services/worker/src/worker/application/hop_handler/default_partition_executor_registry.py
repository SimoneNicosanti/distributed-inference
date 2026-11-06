import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from shared.plan.hop import PlannedExecutableHopId
from shared.plan.plan_version import PlanVersion
from worker.application.hop_handler.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)


class DefaultPartitionExecutorRegistry(PartitionExecutorRegistry):
    @dataclass(frozen=True, slots=True)
    class _PartitionExecutorBundle:
        partition_executor: PartitionExecutor
        lock: asyncio.Lock

    @dataclass(frozen=True, slots=True)
    class _DeploymentKey:
        plan_version: PlanVersion
        hop_id: PlannedExecutableHopId

    def __init__(self) -> None:
        self._partition_executors: dict[
            DefaultPartitionExecutorRegistry._DeploymentKey,
            DefaultPartitionExecutorRegistry._PartitionExecutorBundle,
        ] = {}

    @override
    async def register_partition_executor(
        self,
        plan_version: PlanVersion,
        executable_hop_id: PlannedExecutableHopId,
        partition_executor: PartitionExecutor,
    ) -> None:
        key = self._DeploymentKey(plan_version, executable_hop_id)
        if key in self._partition_executors:
            raise ValueError(
                f"Partition executor already present for hop {executable_hop_id}"
            )

        bundle = DefaultPartitionExecutorRegistry._PartitionExecutorBundle(
            partition_executor, asyncio.Lock()
        )
        self._partition_executors[key] = bundle

    @override
    async def unregister_partition_executor(
        self,
        plan_version: PlanVersion,
        executable_hop_id: PlannedExecutableHopId,
    ) -> PartitionExecutor:
        key = self._DeploymentKey(plan_version, executable_hop_id)
        if key not in self._partition_executors:
            raise KeyError(f"Partition executor not found for hop {executable_hop_id}")
        bundle = self._partition_executors.pop(key)
        async with bundle.lock:
            return bundle.partition_executor

    @override
    @asynccontextmanager
    async def acquire_partition_executor(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> AsyncGenerator[PartitionExecutor]:
        key = self._DeploymentKey(plan_version, executable_hop_id)
        if key not in self._partition_executors:
            raise KeyError(f"Partition executor not found for hop {executable_hop_id}")
        bundle = self._partition_executors[key]

        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.partition_executor

    @override
    async def check_partition_executor_exists(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> bool:
        key = self._DeploymentKey(plan_version, executable_hop_id)
        return key in self._partition_executors
