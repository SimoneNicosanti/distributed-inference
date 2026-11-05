from collections.abc import Mapping
from dataclasses import replace

import pytest

from controller.application.optimization.default_optimization_solution_enricher import (
    DefaultOptimizationSolutionEnricher,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantInfo,
    ModelVariantStaticProfile,
    TensorProperty,
)
from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.model.model import ModelId, ModelTask
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from shared.user.user import UserId


def _variant(model_name: str = "model", tag: str = "selected") -> ModelVariantId:
    return ModelVariantId(ModelId(model_name), tag)


def _solution(*variants: ModelVariantId) -> OptimizationSolution:
    worker_id = ServiceId()
    component = ComponentSolution(
        component_index=1,
        worker_id=worker_id,
        backend=Backend.CPU,
        layer_keys=("input", "first", "second", "output"),
        input_layer_keys=("input",),
        output_layer_keys=("output",),
        tensor_contract=ComponentTensorContract(("features", "mask"), ("prediction",)),
        priority=2,
        mean_execution_time_s=0.1,
        mean_execution_queue_time_s=0.2,
        mean_transmission_time_s=0.3,
        transmission_latency_s=0.4,
    )
    return OptimizationSolution(
        flow_solutions=tuple(
            FlowOptimizationSolution(
                flow_id=FlowId(user_id=UserId()),
                ingress_worker_id=worker_id,
                sink_worker_id=worker_id,
                model_variant_id=variant,
                components=(component,),
                cuts=(),
                mean_batch_fill_time_s=0.5,
                mean_response_time_s=1.5,
            )
            for variant in variants
        ),
        objective_value=1.75,
        is_optimal=False,
    )


def _profile(variant: ModelVariantId) -> ModelVariantStaticProfile:
    return ModelVariantStaticProfile(
        model_variant_id=variant,
        info=ModelVariantInfo(ModelTask.CLASSIFICATION, ()),
        topology=ModelVariantTopology(
            layers={
                "input": LayerDescription("input", "Input", True, False),
                "first": LayerDescription("first", "Compute", False, False),
                "second": LayerDescription("second", "Compute", False, False),
                "output": LayerDescription("output", "Output", False, True),
            },
            edges=(
                EdgeDescription("input", "first", frozenset({"features", "mask"})),
                EdgeDescription("input", "second", frozenset({"features"})),
                EdgeDescription("first", "second", frozenset({"hidden"})),
                EdgeDescription("second", "output", frozenset({"prediction"})),
            ),
        ),
        invariant_layer_properties={},
        invariant_tensor_properties={
            "features": TensorProperty("float32", 4, ("batch", 3)),
            "mask": TensorProperty("bool", 1, ("batch",)),
            "hidden": TensorProperty("float32", 4, ("batch", 3)),
            "prediction": TensorProperty("float32", 4, ("batch", 2)),
        },
        shape_profiles=(),
        optimization_contractions=(),
    )


def test_enriches_only_selected_variants_and_preserves_the_original_solution() -> None:
    first, second, unused = _variant(), _variant("other-model"), _variant(tag="unused")
    solution = _solution(first, second, first)
    profiles = {variant: _profile(variant) for variant in (first, second, unused)}
    profiles[second] = replace(
        profiles[second],
        invariant_tensor_properties={
            **profiles[second].invariant_tensor_properties,
            "features": TensorProperty("float64", 8, ("samples", 7)),
        },
    )

    enriched = DefaultOptimizationSolutionEnricher().enrich_solution(solution, profiles)

    assert enriched.solution is solution
    assert set(enriched.input_tensor_properties_by_variant) == {first, second}
    for variant in (first, second):
        inputs = enriched.input_tensor_properties_by_variant[variant]
        assert set(inputs) == {"features", "mask"}
        assert (
            inputs["features"]
            is profiles[variant].invariant_tensor_properties["features"]
        )
        assert inputs["mask"] == TensorProperty("bool", 1, ("batch",))
    assert enriched.input_tensor_properties_by_variant[first][
        "features"
    ].shape_expression == ("batch", 3)
    assert enriched.input_tensor_properties_by_variant[second][
        "features"
    ] == TensorProperty("float64", 8, ("samples", 7))


def test_empty_solution_requires_no_static_profiles() -> None:
    solution = _solution()

    enriched = DefaultOptimizationSolutionEnricher().enrich_solution(solution, {})

    assert enriched.solution is solution
    assert not enriched.input_tensor_properties_by_variant


def test_rejects_missing_selected_profile() -> None:
    with pytest.raises(ValueError, match="No static profile for selected variant"):
        DefaultOptimizationSolutionEnricher().enrich_solution(_solution(_variant()), {})


def test_rejects_profile_with_a_different_variant_identity() -> None:
    selected = _variant()

    with pytest.raises(ValueError, match="does not match the selected model variant"):
        DefaultOptimizationSolutionEnricher().enrich_solution(
            _solution(selected), {selected: _profile(_variant("other-model"))}
        )


def test_rejects_missing_model_input_properties() -> None:
    selected = _variant()
    profile = _profile(selected)
    properties = dict(profile.invariant_tensor_properties)
    del properties["features"]
    profile = replace(profile, invariant_tensor_properties=properties)

    with pytest.raises(ValueError, match=r"missing input tensor properties:.*features"):
        DefaultOptimizationSolutionEnricher().enrich_solution(
            _solution(selected), {selected: profile}
        )


def test_rejects_topology_without_model_input_tensors() -> None:
    selected = _variant()
    profile = _profile(selected)
    layers = dict(profile.topology.layers())
    layers["input"] = replace(layers["input"], is_input=False)
    profile = replace(
        profile,
        topology=ModelVariantTopology(
            layers=layers, edges=profile.topology.edges().values()
        ),
    )

    with pytest.raises(ValueError, match="has no model input tensors"):
        DefaultOptimizationSolutionEnricher().enrich_solution(
            _solution(selected), {selected: profile}
        )


def test_enriched_solution_copies_both_levels_of_input_mappings() -> None:
    selected = _variant()
    property_ = TensorProperty("float32", 4, ("batch", 3))
    inputs = {"features": property_}
    by_variant: dict[ModelVariantId, Mapping[TensorKey, TensorProperty]] = {
        selected: inputs
    }
    enriched = EnrichedOptimizationSolution(_solution(selected), by_variant)

    inputs.clear()
    by_variant.clear()

    assert enriched.input_tensor_properties_by_variant == {
        selected: {"features": property_}
    }


@pytest.mark.parametrize("variant_tags", [(), ("selected", "unused")])
def test_enriched_solution_requires_exactly_the_selected_variants(
    variant_tags: tuple[str, ...],
) -> None:
    by_variant: dict[ModelVariantId, Mapping[TensorKey, TensorProperty]] = {
        _variant(tag=tag): {"features": TensorProperty("float32", 4, (1, 3))}
        for tag in variant_tags
    }

    with pytest.raises(ValueError, match="must match the selected model variants"):
        EnrichedOptimizationSolution(_solution(_variant()), by_variant)


def test_enriched_solution_rejects_empty_input_properties() -> None:
    selected = _variant()

    with pytest.raises(ValueError, match="must have input tensor properties"):
        EnrichedOptimizationSolution(_solution(selected), {selected: {}})


def test_enriched_solution_rejects_duplicate_flows() -> None:
    selected = _variant()
    solution = _solution(selected)
    solution = replace(solution, flow_solutions=solution.flow_solutions * 2)

    with pytest.raises(
        ValueError, match="Flow ids in an enriched solution must be unique"
    ):
        DefaultOptimizationSolutionEnricher().enrich_solution(
            solution, {selected: _profile(selected)}
        )
