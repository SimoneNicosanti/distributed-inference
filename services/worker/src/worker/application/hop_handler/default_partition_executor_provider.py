import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import override

from shared.plan.hop import PlannedExecutableHopId
from shared.plan.plan_version import PlanVersion
from worker.application.hop_handler.abc.hop_executor_initializer import (
    HopExecutorInitializer,
)
from worker.application.hop_handler.abc.hop_executor_provider import (
    HopExecutorProvider,
)
from worker.application.ports.outbound.hop_execution.hop_executor import (
    HopExecutor,
)


class DefaultPartitionExecutorProvider(HopExecutorProvider):
    @dataclass(frozen=True, slots=True)
    class _PartitionExecutorBundle:
        partition_executor: HopExecutor
        lock: asyncio.Lock

    @dataclass(frozen=True, slots=True)
    class _DeploymentKey:
        plan_version: PlanVersion
        hop_id: PlannedExecutableHopId

    def __init__(self, partition_executor_initializer: HopExecutorInitializer) -> None:
        self._partition_executor_initializer = partition_executor_initializer

        self._partition_executors: dict[
            DefaultPartitionExecutorProvider._DeploymentKey,
            DefaultPartitionExecutorProvider._PartitionExecutorBundle,
        ] = {}

        self._init_tasks: dict[
            DefaultPartitionExecutorProvider._DeploymentKey, asyncio.Task[None]
        ] = {}

    @override
    @asynccontextmanager
    async def acquire_hop_executor(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> AsyncGenerator[HopExecutor]:
        key = self._DeploymentKey(plan_version, executable_hop_id)
        if key not in self._partition_executors:
            await self._wait_for_executor_init(key)

        bundle = self._partition_executors[key]
        ## We do a yield while keeping the lock -> No other task can acquire the lock
        async with bundle.lock:
            yield bundle.partition_executor

    async def _wait_for_executor_init(
        self,
        key: DefaultPartitionExecutorProvider._DeploymentKey,
    ) -> None:
        task = self._init_tasks.get(key)

        if task is None:
            task = asyncio.create_task(self._initialize_and_publish(key))
            self._init_tasks[key] = task
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )

        await asyncio.shield(task)

    async def _initialize_and_publish(
        self,
        key: DefaultPartitionExecutorProvider._DeploymentKey,
    ) -> None:
        try:
            executor = await self._partition_executor_initializer.initialize(
                key.plan_version, key.hop_id
            )
            self._partition_executors[key] = self._PartitionExecutorBundle(
                executor, asyncio.Lock()
            )
        finally:
            self._init_tasks.pop(key, None)
