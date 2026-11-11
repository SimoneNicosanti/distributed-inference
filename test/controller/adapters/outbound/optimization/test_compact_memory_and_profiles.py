from dataclasses import replace

import pytest
from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    OptimizationInfeasibleError,
)
from test.controller.adapters.outbound.optimization.test_compact_formulation import (
    _context,
)
from test.controller.adapters.outbound.optimization.test_gurobi_solver import (
    _flow,
    _problem_config,
    _request,
    _worker,
)
from test.model_manager.test_profile_contract import (
    profiled_variant as profiled_variant,
)

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.profile_bounds import (
    cut_upper_mb,
)
from controller.adapters.outbound.profile.mappers.model_variant_static_profile_mapper import (
    ModelVariantStaticProfileMapper,
)
from controller.application.optimization.formulation.compact.profile_data import (
    ProfileDataError,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_config import (
    FormulationKind,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_static_profile import (
    LayerProperty,
    TensorShapeProperty,
    WeightProperty,
)
from integration.dto.model.model_dto import ModelDto
from model_manager.adapters.outbound.model_metadata_store.mappers.profiled_model_variant_mapper import (
    ProfiledModelVariantMapper,
)
from shared.model.model import CNNArchitectureInfo, ModelTask


def test_maximum_cut_survives_existing_profiler_dto_controller_chain(profiled_variant):
    dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(profiled_variant)
    mapped = ModelVariantStaticProfileMapper.model_variant_static_profile_from_dto(
        dto,
        ModelDto(
            model_id=profiled_variant.model_variant.model_variant_id.model_id,
            model_task=ModelTask.CLASSIFICATION,
            architecture_info=CNNArchitectureInfo(),
        ),
    )
    for shape, original in zip(
        mapped.shape_profiles, dto.profile.variable_properties, strict=True
    ):
        assert shape.max_cut_size_mib == original.model_properties.max_cut_size_mib
        single = replace(mapped, shape_profiles=(shape,))
        assert cut_upper_mb(single) == pytest.approx(
            original.model_properties.max_cut_size_mib * 1024**2 / 1e6
        )


def test_compact_rejects_missing_profiled_cut_instead_of_inventing_a_bound():
    profile = next(
        iter(_context({_worker(1): 0.1}).static_profiles_by_variant.values())
    )
    profile = replace(
        profile,
        shape_profiles=tuple(
            replace(shape, max_cut_size_mib=None) for shape in profile.shape_profiles
        ),
    )
    with pytest.raises(ProfileDataError, match="profiled max_cut_size_mib"):
        cut_upper_mb(profile)


def context_with_memory(*, capacity, weights=False):
    worker = _worker(1)
    context = _context({worker: 0.1})
    profiles = {}
    for variant, profile in context.static_profiles_by_variant.items():
        layers = dict(profile.invariant_layer_properties)
        if weights:
            layers["compute"] = LayerProperty(
                weight_properties={"weight": WeightProperty("float32", (250000,), 4)}
            )
        profiles[variant] = replace(
            profile,
            invariant_layer_properties=layers,
            shape_profiles=tuple(
                replace(
                    shape,
                    max_cut_size_mib=1e6 / 1024**2,
                    tensor_properties={
                        tensor: TensorShapeProperty((250000,))
                        for tensor in shape.tensor_properties
                    },
                )
                for shape in profile.shape_profiles
            ),
        )
    resources = {
        worker: replace(
            context.resource_profiles_by_worker[worker],
            ram_profile=replace(
                context.resource_profiles_by_worker[worker].ram_profile,
                capacity_bytes=capacity,
                available_bytes=capacity,
            ),
        )
    }
    return OptimizationContext(
        static_profiles_by_variant=profiles,
        network_topology=context.network_topology,
        resource_profiles_by_worker=resources,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
    )


@pytest.mark.parametrize("capacity", [2_900_000, 3_100_000])
def test_resident_weights_sum_across_flows_but_activation_is_maximum(capacity):
    worker = _worker(1)
    context = context_with_memory(capacity=capacity, weights=True)
    requests = {
        r.flow_id: r
        for r in (
            _request(worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(i))
            for i in (1, 2)
        )
    }
    solver = GurobiSolver(
        GurobiSolverConfig(
            CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            formulation=FormulationKind.COMPACT,
        )
    )
    if capacity < 3_000_000:
        with pytest.raises(OptimizationInfeasibleError):
            solver._solve(requests, context, _problem_config(1))
    else:
        assert solver._solve(requests, context, _problem_config(1)).is_optimal


def test_profiled_cut_activation_fits_despite_larger_layer_activation_sum():
    worker = _worker(1)
    context = context_with_memory(capacity=1_500_000)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )

    result = GurobiSolver(
        GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    )._solve({request.flow_id: request}, context, _problem_config(1))
    assert result.is_optimal
