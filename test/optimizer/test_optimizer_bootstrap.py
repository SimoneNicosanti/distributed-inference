from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from redis import asyncio as redis_asyncio

from advertisement.advertising_manager import AdvertisingManager
from advertisement.http.http_capability_life_checker import (
    DEFAULT_HTTP_READINESS_PATH,
    build_http_capability_life_checkers,
)
from catalog.api.service.service_registrar import ServiceRegistrar
from integration.optimizer.http.optimizer_http_paths import OPTIMIZE_PATH
from integration.service_instance.service_instance import (
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.application.optimization_controller import DefaultOptimizerController
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.application.strategies.full_solver_strategy import FullSolverStrategy
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.incremental.persistent_incremental_flow_solver_strategy import (
    PersistentIncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind
from optimizer.composition.default_composition_root import DefaultCompositionRoot
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.main import build_http_app, build_service_instance, main


@pytest.mark.parametrize("strategy_kind", list(SolverStrategyKind))
def test_builds_optimizer_owned_strategy(strategy_kind: SolverStrategyKind) -> None:
    root = DefaultCompositionRoot(
        solver_config=GurobiSolverConfig(time_limit_s=7, threads=1),
        formulation_config=MisocpCompactFormulationConfig(
            relative_error=0.2,
            representation=CompactFormulationKind.CUMULATIVE,
        ),
        strategy_config=GurobiStrategyConfig(strategy=strategy_kind),
    )
    controller = root.build_optimizer_controller()
    assert isinstance(controller, DefaultOptimizerController)
    strategy = controller._strategy
    if strategy_kind == SolverStrategyKind.FULL:
        assert isinstance(strategy, FullSolverStrategy)
        assert strategy._solver_config.time_limit_s == 7
    else:
        if strategy_kind == SolverStrategyKind.INCREMENTAL_PERSISTENT:
            assert isinstance(strategy, PersistentIncrementalFlowSolverStrategy)
            strategy = strategy._strategy
        assert isinstance(strategy, IncrementalFlowSolverStrategy)
        assert strategy._config.insertion_solver_config.time_limit_s == 7
    assert (
        strategy._formulation_config.representation == CompactFormulationKind.CUMULATIVE
    )


@pytest.mark.asyncio
async def test_readiness_advertises_optimization_at_the_external_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ID", "optimizer-test")
    monkeypatch.setenv("OPTIMIZER_ADVERTISED_HOST", "optimizer.example")
    monkeypatch.setenv("OPTIMIZER_HTTP_PORT", "8100")
    instance = build_service_instance()
    controller = AsyncMock(spec=OptimizerController)
    app = build_http_app(controller)
    registrar = AsyncMock(spec=ServiceRegistrar)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        checkers = build_http_capability_life_checkers(instance, client)
        manager = AdvertisingManager(registrar, instance.service_id, checkers, 20)
        await manager._advertise()
        response = await client.get(
            f"http://127.0.0.1:8100{DEFAULT_HTTP_READINESS_PATH}"
        )
        assert response.status_code == 204
        assert response.content == b""
        invalid = await client.post(f"http://127.0.0.1:8100{OPTIMIZE_PATH}", json={})
        assert invalid.status_code == 422
    controller.optimize.assert_not_awaited()
    advertised = registrar.register_or_renew.await_args.args[0]
    assert advertised.service_id.id == "optimizer-test"
    assert len(advertised.capabilities) == 1
    capability = advertised.capabilities[0]
    assert capability.type == CapabilityType.OPTIMIZATION
    assert capability.interfaces[0].protocol == CapabilityProtocol.HTTP
    assert str(capability.interfaces[0].endpoint) == "http://optimizer.example:8100/"
    assert checkers[0]._probe_url == "http://127.0.0.1:8100/health/ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_composition_stops_advertising_and_closes_redis(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://test-redis:6379/2")
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.ping = AsyncMock(return_value=True)
    redis.__aenter__.return_value = redis
    advertising = AsyncMock(spec=AdvertisingManager)
    with (
        patch(
            "optimizer.composition.default_composition_root.redis_asyncio.Redis.from_url",
            return_value=redis,
        ) as from_url,
        patch(
            "optimizer.composition.default_composition_root.AdvertisingManager",
            return_value=advertising,
        ),
    ):
        try:
            async with DefaultCompositionRoot().compose(
                build_service_instance()
            ) as controller:
                assert isinstance(controller, DefaultOptimizerController)
                from_url.assert_called_once_with("redis://test-redis:6379/2")
                redis.ping.assert_awaited_once()
                advertising.start.assert_awaited_once()
                assert not redis.__aexit__.called
                if fail:
                    raise RuntimeError("request failed")
        except RuntimeError as error:
            assert fail and str(error) == "request failed"
        advertising.stop.assert_awaited_once()
        redis.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_releases_composition_when_server_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPTIMIZER_HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("OPTIMIZER_HTTP_PORT", "8100")
    controller = AsyncMock(spec=OptimizerController)
    closed: list[bool] = []

    @asynccontextmanager
    async def compose(instance: ServiceInstance) -> AsyncGenerator[OptimizerController]:
        assert instance.capabilities[0].type == CapabilityType.OPTIMIZATION
        try:
            yield controller
        finally:
            closed.append(True)

    server = AsyncMock()
    server.serve.side_effect = RuntimeError("server failed")
    with (
        patch("optimizer.main.DefaultCompositionRoot") as root,
        patch("optimizer.main.uvicorn.Server", return_value=server) as server_factory,
    ):
        root.return_value.compose.side_effect = compose
        with pytest.raises(RuntimeError, match="server failed"):
            await main()
        config = server_factory.call_args.args[0]
        assert config.host == "127.0.0.1"
        assert config.port == 8100
        assert config.app.title == "Optimizer"
    assert closed == [True]
