import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.abc.solution_to_plan_converter import (
    SolutionToPlanConverter,
)
from controller.application.deployment.default_actuation_manager import (
    DefaultActuationManager,
)
from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.application.ports.outbound.plan_version.plan_version_factory import (
    PlanVersionFactory,
)
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.plan.worker_plan import WorkerPlan
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId, WorkerId


@dataclass
class _Setup:
    manager: DefaultActuationManager
    observer: AsyncMock
    gateway: AsyncMock
    publisher: AsyncMock
    solution: EnrichedOptimizationSolution
    workers: tuple[WorkerId, WorkerId]


@pytest.fixture
def setup() -> _Setup:
    observer = AsyncMock(spec=SolutionObserver)
    gateway = AsyncMock(spec=PlanShareGateway)
    publisher = AsyncMock(spec=FlowPlanPublisher)
    workers = (ServiceId(), ServiceId())
    converter = AsyncMock(spec=SolutionToPlanConverter)
    converter.convert_solution_to_plans.return_value = (
        {worker: Mock(spec=WorkerPlan) for worker in workers},
        {},
    )
    manager = DefaultActuationManager(
        solution_observer=observer,
        partition_builder_gateway=AsyncMock(spec=PartitionBuilderGateway),
        plan_version_factory=AsyncMock(
            spec=PlanVersionFactory,
            build_next_plan_version=AsyncMock(return_value=PlanVersion(1)),
        ),
        solution_to_plan_converter=converter,
        plan_share_gateway=gateway,
        flow_plan_publisher=publisher,
    )
    return _Setup(
        manager,
        observer,
        gateway,
        publisher,
        EnrichedOptimizationSolution(OptimizationSolution((), 0, True), {}),
        workers,
    )


@pytest.mark.asyncio
async def test_success_waits_for_all_workers_at_both_phase_boundaries(
    setup: _Setup,
) -> None:
    deploy_started, apply_started = asyncio.Event(), asyncio.Event()
    release_deploy, release_apply = asyncio.Event(), asyncio.Event()

    async def deploy(worker: WorkerId, plan: object) -> None:
        if worker == setup.workers[1]:
            deploy_started.set()
            await release_deploy.wait()

    async def apply(worker: WorkerId, version: object) -> None:
        if worker == setup.workers[1]:
            apply_started.set()
            await release_apply.wait()

    setup.gateway.send_prepare_plan_message.side_effect = deploy
    setup.gateway.send_apply_plan_message.side_effect = apply
    task = asyncio.create_task(setup.manager.actuate_solution(setup.solution))
    try:
        async with asyncio.timeout(1):
            await deploy_started.wait()
            setup.gateway.send_apply_plan_message.assert_not_awaited()
            setup.publisher.publish_flow_plans.assert_not_awaited()
            release_deploy.set()
            await apply_started.wait()
            setup.publisher.publish_flow_plans.assert_not_awaited()
            release_apply.set()
            await task
    finally:
        release_deploy.set()
        release_apply.set()
        await asyncio.gather(task, return_exceptions=True)

    assert setup.gateway.send_prepare_plan_message.await_count == len(setup.workers)
    assert setup.gateway.send_apply_plan_message.await_count == len(setup.workers)
    setup.publisher.publish_flow_plans.assert_awaited_once_with({})


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["deploy", "apply"])
@pytest.mark.parametrize("error_type", [RuntimeError, httpx.ReadTimeout])
async def test_failure_waits_for_remaining_workers_and_stops_later_phases(
    setup: _Setup,
    phase: str,
    error_type: type[Exception],
) -> None:
    failed, slow_started, release, finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def operation(worker: WorkerId, argument: object) -> None:
        if worker == setup.workers[0]:
            failed.set()
            raise error_type("worker failed")
        slow_started.set()
        await release.wait()
        finished.set()

    gateway_method = (
        setup.gateway.send_prepare_plan_message
        if phase == "deploy"
        else setup.gateway.send_apply_plan_message
    )
    gateway_method.side_effect = operation
    task = asyncio.create_task(setup.manager.actuate_solution(setup.solution))
    try:
        async with asyncio.timeout(1):
            await failed.wait()
            await slow_started.wait()
            assert not task.done()
            setup.publisher.publish_flow_plans.assert_not_awaited()
            if phase == "deploy":
                setup.gateway.send_apply_plan_message.assert_not_awaited()
            release.set()
            with pytest.raises(
                ExceptionGroup, match=f"Worker {phase} failed"
            ) as caught:
                await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert finished.is_set()
    error = caught.value.exceptions[0]
    assert isinstance(error, error_type)
    assert any(str(setup.workers[0].id) in note for note in error.__notes__)
    assert any("plan version 1" in note for note in error.__notes__)
    setup.publisher.publish_flow_plans.assert_not_awaited()


@pytest.mark.asyncio
async def test_reports_all_worker_failures(setup: _Setup) -> None:
    async def deploy(worker: WorkerId, plan: object) -> None:
        raise RuntimeError(str(worker.id))

    setup.gateway.send_prepare_plan_message.side_effect = deploy

    with pytest.raises(ExceptionGroup) as caught:
        await setup.manager.actuate_solution(setup.solution)

    assert {str(error) for error in caught.value.exceptions} == {
        str(worker.id) for worker in setup.workers
    }
    setup.gateway.send_apply_plan_message.assert_not_awaited()
    setup.publisher.publish_flow_plans.assert_not_awaited()


@pytest.mark.asyncio
async def test_publication_failure_is_propagated_after_worker_apply(
    setup: _Setup,
) -> None:
    setup.publisher.publish_flow_plans.side_effect = RuntimeError("publication failed")

    with pytest.raises(RuntimeError, match="publication failed") as caught:
        await setup.manager.actuate_solution(setup.solution)

    assert setup.gateway.send_apply_plan_message.await_count == len(setup.workers)
    assert any(
        "publication failed for plan version 1" in note
        for note in caught.value.__notes__
    )


@pytest.mark.asyncio
async def test_cancellation_does_not_apply_or_publish(setup: _Setup) -> None:
    started, finished = asyncio.Event(), asyncio.Event()

    async def deploy(worker: WorkerId, plan: object) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    setup.gateway.send_prepare_plan_message.side_effect = deploy
    task = asyncio.create_task(setup.manager.actuate_solution(setup.solution))
    try:
        async with asyncio.timeout(1):
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert finished.is_set()
    setup.gateway.send_apply_plan_message.assert_not_awaited()
    setup.publisher.publish_flow_plans.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_reports_failure_and_processes_the_next_solution(
    setup: _Setup,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup.observer.observe_new_solution.side_effect = [
        setup.solution,
        setup.solution,
        asyncio.CancelledError(),
    ]
    with patch.object(
        setup.manager, "actuate_solution", new_callable=AsyncMock
    ) as actuate:
        actuate.side_effect = [RuntimeError("actuation failed"), None]
        with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
            await setup.manager._observation_loop()

        assert actuate.await_count == 2
    assert "Solution actuation failed" in caplog.text
