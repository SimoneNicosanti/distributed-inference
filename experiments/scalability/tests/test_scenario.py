import math
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
from scalability_benchmark import instances
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.catalog import (
    CatalogEntry,
    selection_order,
    synthetic_accuracies,
)
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import Scenario, build_scenario, make_instance
from scalability_benchmark.network import select_network
from scalability_benchmark.profiling.static import StaticModel
from scalability_benchmark.randomness import fingerprint

from controller.adapters.outbound.optimization.gurobi_problem_builder import (
    DefaultGurobiProblemBuilder,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.solving.solver_config import ProblemConfig
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


@pytest_asyncio.fixture
async def scenario(
    config: ExperimentConfig,
    make_model: Callable[[CatalogEntry], StaticModel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Scenario:
    settings = {model.name: model for model in config.catalog.models}
    metrics = synthetic_accuracies(config, 42)
    entries = tuple(
        CatalogEntry(
            settings[name],
            tmp_path / settings[name].filename,
            ModelVariantId(ModelId(name), "fp32-batch1"),
            metrics[name],
            {settings[name].filename: "mock-content-hash"},
        )
        for name in selection_order(config, 42)
    )

    async def fake_profile(
        entry: CatalogEntry, cache: Path, software_hash: str
    ) -> StaticModel:
        return make_model(entry)

    monkeypatch.setattr(instances, "discover_catalog", lambda config, seed: entries)
    monkeypatch.setattr(instances, "profile_entry", fake_profile)
    first = await build_scenario(config, 42, tmp_path, "test-software")
    second = await build_scenario(config, 42, tmp_path, "test-software")
    assert fingerprint(first.as_record()) == fingerprint(second.as_record())
    return first


@pytest.mark.asyncio
async def test_nested_entities_and_requests(scenario: Scenario) -> None:
    small_spec = RunSpec("main", 42, 0.6, 1, 1, 0, 0, 300)
    large_spec = RunSpec("main", 42, 0.6, 5, 6, 5, 5, 300)
    small_requests, small = make_instance(scenario, small_spec)
    requests, large = make_instance(scenario, large_spec)
    assert small_requests.items() <= requests.items()
    assert all(
        large.static_profiles_by_variant[key] == value
        for key, value in small.static_profiles_by_variant.items()
    )
    for worker, resource in small.resource_profiles_by_worker.items():
        assert large.resource_profiles_by_worker[worker] == resource
        for model, execution in small.execution_profiles_by_worker[
            worker
        ].execution_profiles.items():
            assert (
                large.execution_profiles_by_worker[worker].execution_profiles[model]
                == execution
            )
    assert len(small.network_topology.connections()) == 0
    assert len(large.network_topology.connections()) == 110
    mid_workers = scenario.workers[:6]
    mid = select_network(mid_workers, scenario.links)
    assert mid.connections().items() <= large.network_topology.connections().items()
    device = scenario.workers[0].id
    for request in requests.values():
        assert request.flow_endpoint_placement.ingress_placement == FixedPlacement(
            device
        )
        assert request.flow_endpoint_placement.sink_placement == FixedPlacement(device)
        assert request.local_worker_config == ValidLocalWorkerConfig(device, True)
        assert 0.2 <= request.qos_requirements.max_mean_response_time <= 5
    problem = DefaultGurobiProblemBuilder().build(
        requests,
        large,
        ProblemConfig(**scenario.config.optimization.model_dump(exclude={"objective"})),
    )
    assert all(len(models) == 6 for models in problem.models_by_flow.values())
    assert device in problem.workers


@pytest.mark.asyncio
async def test_homogeneity_projection_and_calibration(scenario: Scenario) -> None:
    for tier, capacity in (("device", 8), ("edge", 8), ("cloud", 24)):
        workers = [worker for worker in scenario.workers if worker.tier == tier]
        assert len(workers) == (1 if tier == "device" else 5)
        assert all(
            worker.resource.ram_profile.available_bytes == capacity * 2**30
            for worker in workers
        )
        assert all(worker.hardware == workers[0].hardware for worker in workers)
    for load, flows in scenario.flows.items():
        assert math.isclose(
            sum(flow.stream_info.arrival_rate for flow in flows)
            * scenario.reference_time_s,
            load,
        )
    low, high = scenario.flows[0.3], scenario.flows[0.8]
    for left, right in zip(low, high, strict=True):
        assert left.flow_id == right.flow_id
        assert left.qos_requirements == right.qos_requirements
        assert math.isclose(
            left.stream_info.arrival_rate / right.stream_info.arrival_rate, 0.3 / 0.8
        )
    for link in scenario.links:
        reverse = next(
            other
            for other in scenario.links
            if other.source == link.target and other.target == link.source
        )
        assert link.throughput_bytes_per_sec == reverse.throughput_bytes_per_sec
        assert link.round_trip_time_s == reverse.round_trip_time_s
