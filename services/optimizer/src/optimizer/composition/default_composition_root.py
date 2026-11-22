import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import override

import httpx
from redis import asyncio as redis_asyncio

from advertisement.advertising_manager import AdvertisingManager
from advertisement.http.http_capability_life_checker import (
    build_http_capability_life_checkers,
)
from catalog.redis.service.redis_service_registrar import RedisServiceRegistrar
from integration.service_instance.service_instance import ServiceInstance
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.adapters.outbound.gurobi.strategy_factory import (
    build_compact_gurobi_solver_strategy,
)
from optimizer.application.optimization_controller import DefaultOptimizerController
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.composition.composition_root import CompositionRoot
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from redis_access.address_factory.service.service_instance_address_factory import (
    HashMapServiceInstanceAddressFactory,
)

DEFAULT_REDIS_URL = "redis://redis:6379/0"


class DefaultCompositionRoot(CompositionRoot):
    def __init__(
        self,
        solver_config: GurobiSolverConfig | None = None,
        formulation_config: MisocpCompactFormulationConfig | None = None,
        strategy_config: GurobiStrategyConfig | None = None,
    ) -> None:
        self._solver_config = solver_config or GurobiSolverConfig(
            output_flag=True,
            time_limit_s=180,
        )
        self._formulation_config = formulation_config or MisocpCompactFormulationConfig(
            relative_error=0.1,
        )
        self._strategy_config = strategy_config or GurobiStrategyConfig()

    @asynccontextmanager
    @override
    async def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[OptimizerController]:
        optimizer_controller = self.build_optimizer_controller()
        async with AsyncExitStack() as stack:
            redis = await stack.enter_async_context(
                self.build_redis_connection_context()
            )
            await stack.enter_async_context(
                self.build_advertising_manager_context(redis, service_instance)
            )
            yield optimizer_controller

    def build_optimizer_controller(self) -> OptimizerController:
        strategy = build_compact_gurobi_solver_strategy(
            self._formulation_config,
            self._solver_config,
            strategy_config=self._strategy_config,
        )
        return DefaultOptimizerController(strategy)

    @asynccontextmanager
    async def build_redis_connection_context(
        self,
    ) -> AsyncGenerator[redis_asyncio.Redis]:
        async with redis_asyncio.Redis.from_url(
            os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        ) as redis:
            await redis.ping()
            yield redis

    @asynccontextmanager
    async def build_advertising_manager_context(
        self,
        redis: redis_asyncio.Redis,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[AdvertisingManager]:
        async with httpx.AsyncClient() as http_client:
            registrar = RedisServiceRegistrar(
                redis,
                entry_ttl_s=60,
                address_factory=HashMapServiceInstanceAddressFactory(),
            )
            advertising_manager = AdvertisingManager(
                registrar,
                service_instance.service_id,
                build_http_capability_life_checkers(service_instance, http_client),
                heartbeat_interval_s=20,
            )
            await advertising_manager.start()
            try:
                yield advertising_manager
            finally:
                await advertising_manager.stop()
