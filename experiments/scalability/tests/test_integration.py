"""Opt-in integration checks. Never execute during static-only verification."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from scalability_benchmark.catalog import (
    CatalogEntry,
    artifact_hashes,
    synthetic_accuracies,
)
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.flows import flow_pool
from scalability_benchmark.hardware import worker_pool
from scalability_benchmark.network import select_network
from scalability_benchmark.profiling.execution import execution_profile
from scalability_benchmark.profiling.static import (
    StaticModel,
    profile_entry,
)
from scalability_benchmark.solver import build_solver_strategy

from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_execution_profile import (
    WorkerExecutionProfiles,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.application.abc.solver_strategy import SolveError
from optimizer.application.formulation.compact.compact_formulation_config import (
    CompactFormulationConfig,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("SCALABILITY_INTEGRATION") != "1",
        reason="Set SCALABILITY_INTEGRATION=1 to run real profiler/solver integration",
    ),
]


@pytest.mark.asyncio
async def test_real_onnx_static_profiler(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "small.onnx"
    graph = helper.make_graph(
        [helper.make_node("Relu", ["images"], ["logits"], name="relu")],
        "small",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, 224, 224])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.save(model, str(path))
    specification = config.catalog.models[0]
    entry = CatalogEntry(
        specification,
        path,
        ModelVariantId(ModelId("small"), "fp32-batch1"),
        synthetic_accuracies(config, 42)[specification.name],
        artifact_hashes(path),
    )
    result = await profile_entry(entry, tmp_path / "profiles", "integration")
    assert result.profile.shape_profiles[0].tensor_properties
    assert result.profile.shape_profiles[0].max_cut_size_mib == pytest.approx(
        1 * 3 * 224 * 224 * 4 / 1024**2
    )
    assert result.profile.topology.layers()["relu"].type == "Relu"


@pytest.mark.asyncio
async def test_real_solver_feasible_and_infeasible(
    config: ExperimentConfig,
    make_model: Callable[[CatalogEntry], StaticModel],
    tmp_path: Path,
) -> None:
    specification = config.catalog.models[0]
    entry = CatalogEntry(
        specification,
        tmp_path / "mock.onnx",
        ModelVariantId(ModelId(specification.name), "fp32-batch1"),
        synthetic_accuracies(config, 42)[specification.name],
        {},
    )
    model = make_model(entry)
    device = worker_pool(config, 42)[0]
    execution = execution_profile(
        model.profile, config.hardware["device"], config.execution, 42, "device"
    ).profile
    context = OptimizationContext(
        static_profiles_by_variant={entry.variant_id: model.profile},
        network_topology=select_network((device,), ()),
        resource_profiles_by_worker={device.id: device.resource},
        execution_profiles_by_worker={
            device.id: WorkerExecutionProfiles(device.id, {entry.variant_id: execution})
        },
    )
    flow = flow_pool(config, 42, 0.6, 1.0, device.id)[0]
    observed: list[GurobiRunStatistics] = []
    formulation_config = CompactFormulationConfig(
        max_components=7,
        priority_count=3,
        compute_utilization_limit=0.8,
        network_utilization_limit=0.8,
        relative_error=0.1,
        objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
    )
    strategy, recorder = build_solver_strategy(
        formulation_config,
        GurobiSolverConfig(
            time_limit_s=300,
            mip_gap=0.01,
            observer=observed.append,
        ),
    )
    result = await strategy.solve({flow.flow_id: flow}, context)
    assert result.flow_solutions and observed[-1].solution_count > 0
    from dataclasses import replace

    impossible = replace(
        flow,
        qos_requirements=replace(flow.qos_requirements, max_mean_response_time=1e-12),
    )
    with pytest.raises(SolveError):
        await strategy.solve({impossible.flow_id: impossible}, context)
    assert recorder.output is not None
    assert recorder.output.termination is SolverTerminationKind.INFEASIBLE
