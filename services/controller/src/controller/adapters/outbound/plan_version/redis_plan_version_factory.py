from typing import override

from redis import WatchError
from redis import asyncio as redis_asyncio
from redis.asyncio.client import Pipeline

from controller.application.ports.outbound.plan_version.plan_version_factory import (
    PlanVersionFactory,
)
from shared.plan.plan_version import PlanVersion


class RedisPlanVersionFactory(PlanVersionFactory):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        super().__init__()
        self._redis = redis

    @override
    async def build_next_plan_version(self) -> PlanVersion:
        key = self._plan_version_key()

        while True:
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    return await self._optimistic_transaction(key, pipe)

                except WatchError:
                    continue

    async def _optimistic_transaction(self, key: str, pipe: Pipeline) -> PlanVersion:
        await pipe.watch(key)

        value = await pipe.get(key)
        curr_ver_num = 0 if value is None else int(value)
        updated_ver_num = curr_ver_num + 1

        pipe.multi()
        pipe.set(key, updated_ver_num)
        await pipe.execute()

        return PlanVersion(version_number=updated_ver_num)

    def _plan_version_key(self) -> str:
        return "current-plan-version"
