from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from gurobipy import GRB
from test.controller.adapters.inbound.test_http_optimization_manager import (
    RecordingOptimizationManager,
    _client,
    _request_dto,
)
from test.controller.application.optimization.test_compact_optimization_problem_factory import (
    _factory,
    _metric,
    _request,
)
from test.optimizer.test_full_optimizer_runtime import pipeline
from test.optimizer.test_single_configuration_contract import (
    single_configuration_problem,
    single_shape_context,
)

from controller.adapters.outbound.optimizer.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as ControllerMapper,
)
from controller.application.optimization.formulation.compact_optimization_problem_factory import (
    CompactOptimizationProblemBuildError,
)
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.request.model_selection_requirements import (
    ModelSelectionRequirements,
    ModelVariantSelectionStrategy,
    PerformanceSelectionStrategy,
)
from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.dto.optimization.optimization_dto import (
    ModelSelectionRequirementsDto,
    ModelVariantSelectionStrategyDto,
)
from integration.dto.optimizer.optimization_problem_dto import FlowDataDto
from integration.optimizer.http.optimizer_http_schema import OptimizeRequest
from optimizer.adapters.inbound.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as OptimizerMapper,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.objective_builder import (
    CompactObjectiveBuilder,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from shared.flow.flow import FlowId
from shared.model.model import ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)


def test_http_batch_preserves_both_selection_strategies():
    manager = RecordingOptimizationManager()
    performance = _request_dto(FlowId())
    variant = ModelVariantId(ModelId("chosen"), "fp32")
    fixed = _request_dto(FlowId()).model_copy(
        update={
            "model_selection_requirements": ModelSelectionRequirementsDto(
                selection_strategy=ModelVariantSelectionStrategyDto(
                    model_variant_id=variant
                )
            )
        }
    )
    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={
            "requests": [dto.model_dump(mode="json") for dto in (performance, fixed)]
        },
    )
    assert response.status_code == 204
    requests = manager.calls[0]
    assert isinstance(
        requests[performance.flow_id].model_selection_requirements.selection_strategy,
        PerformanceSelectionStrategy,
    )
    assert requests[
        fixed.flow_id
    ].model_selection_requirements == ModelSelectionRequirements(
        ModelVariantSelectionStrategy(variant)
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unknown",
        "both",
        "task_only",
        "variant_only",
        "missing_kind",
        "legacy_qos",
        "legacy_task",
    ],
)
def test_http_rejects_ambiguous_or_incomplete_selection(failure):
    manager = RecordingOptimizationManager()
    payload = _request_dto(FlowId()).model_dump(mode="json")
    strategy = payload["model_selection_requirements"]["selection_strategy"]
    if failure == "missing":
        del payload["model_selection_requirements"]
    elif failure == "unknown":
        strategy["kind"] = "unknown"
    elif failure == "both":
        strategy["model_variant_id"] = {
            "model_id": {"model_name": "chosen"},
            "variant_tag": "fp32",
        }
    elif failure == "task_only":
        del strategy["min_model_performance"]
    elif failure == "variant_only":
        payload["model_selection_requirements"]["selection_strategy"] = {
            "kind": "model_variant"
        }
    elif failure == "missing_kind":
        del strategy["kind"]
    elif failure == "legacy_qos":
        payload["qos_requirements"]["min_model_performance"] = strategy[
            "min_model_performance"
        ]
    else:
        payload["task"] = strategy["task"]
    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH, json={"requests": [payload]}
    )
    assert response.status_code == 422
    assert manager.calls == []


@pytest.mark.asyncio
async def test_mixed_selection_projects_and_transports_only_the_requested_candidates():
    context = single_shape_context()
    profile = next(iter(context.static_profiles_by_variant.values()))
    pinned_id = ModelVariantId(ModelId("pinned"), "fp32")
    pinned = replace(
        profile,
        model_variant_id=pinned_id,
        info=replace(profile.info, task=ModelTask.DETECTION, accuracies=()),
    )
    context = OptimizationContext(
        static_profiles_by_variant={
            profile.model_variant_id: profile,
            pinned_id: pinned,
        },
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
    )
    performance = _request()
    fixed = replace(
        _request(),
        model_selection_requirements=ModelSelectionRequirements(
            ModelVariantSelectionStrategy(pinned_id)
        ),
    )
    original = await _factory().build_optimization_problem(
        {r.flow_id: r for r in (performance, fixed)}, context
    )
    envelope = OptimizeRequest(problem=ControllerMapper.to_dto(original))
    decoded = OptimizeRequest.model_validate_json(envelope.model_dump_json())
    mapped = OptimizerMapper.from_dto(decoded.problem)
    assert {
        key.model_variant_id for key in mapped.flows[performance.flow_id].configurations
    } == {profile.model_variant_id}
    assert mapped.flows[performance.flow_id].performance_requirement == _metric(0.8)
    assert {
        key.model_variant_id for key in mapped.flows[fixed.flow_id].configurations
    } == {pinned_id}
    assert mapped.flows[fixed.flow_id].performance_requirement is None
    assert mapped.variants[pinned_id].accuracies == ()


@pytest.mark.asyncio
async def test_unknown_explicit_variant_does_not_fall_back_to_another_model():
    request = replace(
        _request(),
        model_selection_requirements=ModelSelectionRequirements(
            ModelVariantSelectionStrategy(ModelVariantId(ModelId("missing"), "fp32"))
        ),
    )
    with pytest.raises(
        CompactOptimizationProblemBuildError, match="no eligible model variant"
    ):
        await _factory().build_optimization_problem(
            {request.flow_id: request}, single_shape_context()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,value,eligible",
    [
        (PerformanceMetricType.ACCURACY, 0.8, True),
        (PerformanceMetricType.ACCURACY, 0.95, False),
        (PerformanceMetricType.MAE, 0.95, True),
        (PerformanceMetricType.MAE, 0.8, False),
        (PerformanceMetricType.RMSE, 0.95, True),
        (PerformanceMetricType.RMSE, 0.8, False),
    ],
)
async def test_performance_strategy_preserves_metric_direction(kind, value, eligible):
    context = single_shape_context()
    profile = next(iter(context.static_profiles_by_variant.values()))
    profile = replace(
        profile,
        info=replace(
            profile.info, accuracies=(PerformanceMetric(kind, "dataset", 0.9),)
        ),
    )
    context = OptimizationContext(
        static_profiles_by_variant={profile.model_variant_id: profile},
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
    )
    metric = PerformanceMetric(kind, "dataset", value)
    request = replace(
        _request(),
        model_selection_requirements=ModelSelectionRequirements(
            PerformanceSelectionStrategy(ModelTask.CLASSIFICATION, metric)
        ),
    )
    if eligible:
        problem = await _factory().build_optimization_problem(
            {request.flow_id: request}, context
        )
        assert problem.flows[request.flow_id].performance_requirement == metric
    else:
        with pytest.raises(
            CompactOptimizationProblemBuildError, match="no eligible model variant"
        ):
            await _factory().build_optimization_problem(
                {request.flow_id: request}, context
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["task", "dataset"])
async def test_performance_selection_requires_matching_task_and_dataset(mismatch):
    request = replace(
        _request(),
        model_selection_requirements=ModelSelectionRequirements(
            PerformanceSelectionStrategy(
                ModelTask.DETECTION if mismatch == "task" else ModelTask.CLASSIFICATION,
                replace(_metric(0.8), dataset="other")
                if mismatch == "dataset"
                else _metric(0.8),
            )
        ),
    )
    with pytest.raises(
        CompactOptimizationProblemBuildError, match="no eligible model variant"
    ):
        await _factory().build_optimization_problem(
            {request.flow_id: request}, single_shape_context()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["controller", "transport", "optimizer"])
async def test_absent_performance_requirement_requires_one_configuration(boundary):
    request = _request()
    problem = await _factory().build_optimization_problem(
        {request.flow_id: request}, single_shape_context()
    )
    dto = ControllerMapper.to_dto(problem)
    if boundary == "transport":
        payload = dto.flows[0].model_dump(mode="json")
        payload["performance_requirement"] = None
        payload["configurations"] *= 2
        with pytest.raises(ValueError, match="exactly one configuration"):
            FlowDataDto.model_validate(payload)
    else:
        if boundary == "optimizer":
            problem = OptimizerMapper.from_dto(dto)
        data = next(iter(problem.flows.values()))
        with pytest.raises(ValueError, match="exactly one configuration"):
            replace(
                data,
                performance_requirement=None,
                configurations=data.configurations * 2,
            )


@pytest.mark.parametrize("fixed_only", [False, True])
def test_accuracy_objective_skips_fixed_variants_without_inventing_a_metric(fixed_only):
    problem = single_configuration_problem(2)
    flow, data = next(iter(problem.flows.items()))
    key = data.configurations[0]
    fixed_flow = FlowId()
    fixed = replace(data, performance_requirement=None, configurations=(key,))
    problem = replace(
        problem,
        flows={fixed_flow: fixed, **({} if fixed_only else {flow: data})},
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
        ),
    )
    formulation = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5)
    )
    model = Mock()
    assignment = SimpleNamespace(
        configuration_selected={
            (f, c): int(c == key)
            for f, d in problem.flows.items()
            for c in d.configurations
        }
    )
    CompactObjectiveBuilder().set_objective(
        model, formulation, assignment=assignment, qos=Mock()
    )
    expression, sense = model.setObjective.call_args.args
    assert sense == GRB.MAXIMIZE
    assert expression.getConstant() == pytest.approx(0 if fixed_only else 0.45)


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_native_optimizer_accepts_explicit_variant_without_performance_metrics(
    objective,
):
    problem = single_configuration_problem()
    flow, data = next(iter(problem.flows.items()))
    variant = next(iter(problem.variants))
    problem = replace(
        problem,
        flows={flow: replace(data, performance_requirement=None)},
        variants={variant: replace(problem.variants[variant], accuracies=())},
        policy=replace(problem.policy, objective=objective),
    )
    solution = await pipeline().solve(problem)
    assert solution.is_optimal
    assert solution.flow_solutions[0].model_variant_id == variant
