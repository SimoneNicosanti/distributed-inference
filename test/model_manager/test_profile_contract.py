from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from controller.adapters.outbound.profile.mappers.model_variant_execution_profile_mapper import (
    ModelVariantExecutionProfileMapper,
)
from controller.adapters.outbound.profile.mappers.model_variant_static_profile_mapper import (
    ModelVariantStaticProfileMapper,
)
from controller.application.optimization.formulation.compact.profile_data import (
    batch_size,
    layer_execution_profiles,
    layer_weight_size_bytes,
    tensor_size_bytes,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_execution_profile import (
    WorkerExecutionProfiles,
)
from controller.domain.profile.network_topology import NetworkTopology
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import ProfiledModelVariantDto
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from model_manager.adapters.outbound.model_max_cut_size_computer.pulp_max_cut_size_computer import (
    PulpMaxCutSizeComputer,
)
from model_manager.adapters.outbound.model_metadata_store.mappers.profiled_model_variant_mapper import (
    ProfiledModelVariantMapper,
)
from model_manager.adapters.outbound.model_profiler.onnx_invariant_properties_computer import (
    OnnxInvariantPropertiesComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_model_topology_builder import (
    OnnxModelTopologyBuilder,
)
from model_manager.adapters.outbound.model_profiler.onnx_shape_profile_computer import (
    OnnxShapeProfileComputer,
)
from model_manager.application.default_model_profiler import DefaultModelProfiler
from model_manager.application.ports.outbound.contraction_computer import (
    ContractionComputer,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import CNNArchitectureInfo, ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)
from shared.service.service import ServiceId
from worker.adapters.outbound.profiling.model_execution.ort_engine_model_execution_profiler import (
    OrtModelExecutionProfiler,
)
from worker.adapters.outbound.profiling.store.mappers.model_execution_profile_mapper import (
    ModelExecutionProfileMapper,
)
from worker.adapters.outbound.profiling.store.mappers.model_static_profile_mapper import (
    ModelStaticProfileMapper,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
)
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)


@pytest.fixture
def profiled_variant(tmp_path: Path) -> ProfiledModelVariant:
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["features", "mask"], ["sum"], name="sum"),
                helper.make_node(
                    "MatMul", ["sum", "weights"], ["output"], name="linear"
                ),
            ],
            "contract",
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 3])
                for name in ("features", "mask")
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
            [numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), "weights")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    variant = ModelVariant(
        ModelVariantId(ModelId("contract"), "fp32"),
        ModelVariantInfo(
            precision=NumericPrecision.FP32,
            quantization=QuantizationType.NONE,
            accuracies=(),
            format=ModelVariantFormat.ONNX,
            inputs_info={
                name: InputInfo(("batch", 3)) for name in ("features", "mask")
            },
            dynamic_dimension_values={"batch": DynamicShapeValues(frozenset({1, 2}))},
        ),
    )
    contractions = Mock(spec=ContractionComputer)
    contractions.compute_contractions.return_value = [("sum", "linear")]
    profiler = DefaultModelProfiler(
        OnnxModelTopologyBuilder(),
        OnnxInvariantPropertiesComputer(),
        OnnxShapeProfileComputer(PulpMaxCutSizeComputer()),
        contractions,
    )
    profile = profiler._profile_model_sync(
        ArtifactWorkspace(tmp_path, model_path),
        ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo()),
        variant.model_variant_info,
    )
    return ProfiledModelVariant(variant, profile, ArtifactRef("contract-artifact"))


def test_profile_round_trips_shapes_weights_and_model_properties(
    profiled_variant: ProfiledModelVariant,
) -> None:
    dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(profiled_variant)
    restored = ProfiledModelVariantMapper.from_profiled_model_variant_dto(
        ProfiledModelVariantDto.model_validate_json(dto.model_dump_json())
    )
    assert restored.model_variant == profiled_variant.model_variant
    assert restored.artifact_ref == profiled_variant.artifact_ref
    assert (
        restored.profile.invariable_properties
        == profiled_variant.profile.invariable_properties
    )
    assert (
        restored.profile.variable_properties
        == profiled_variant.profile.variable_properties
    )
    assert (
        restored.profile.topology.edges() == profiled_variant.profile.topology.edges()
    )
    assert restored.profile.optimization_contractions == (("sum", "linear"),)
    for inputs, properties in restored.profile.variable_properties.items():
        batch = inputs.input_shapes["features"][0]
        assert inputs.input_shapes["mask"] == (batch, 3)
        assert properties.tensor_properties["output"].shape == (batch, 2)
        assert properties.model_properties.max_cut_size_mib == pytest.approx(
            batch * 6 * 4 / (1024 * 1024)
        )


def test_rejects_duplicate_configuration_independent_of_input_order(
    profiled_variant: ProfiledModelVariant,
) -> None:
    dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(profiled_variant)
    duplicate = dto.profile.variable_properties[0].model_copy(deep=True)
    duplicate.input_shapes.input_shapes = dict(
        reversed(tuple(duplicate.input_shapes.input_shapes.items()))
    )
    dto.profile.variable_properties.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate input shapes"):
        ProfiledModelVariantMapper.from_profiled_model_variant_dto(dto)


def test_worker_profiles_every_concrete_configuration_on_cpu(
    profiled_variant: ProfiledModelVariant,
    tmp_path: Path,
) -> None:
    dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(profiled_variant)
    static_profile = ModelStaticProfileMapper.model_static_profile_from_dto(dto)
    bundle = ArtifactBundle.from_workspace(
        ArtifactWorkspace(tmp_path, tmp_path / "model.onnx")
    )
    profiler = OrtModelExecutionProfiler()
    for input_shapes in static_profile.input_shape_configurations:
        profile = profiler._sync_profile_model_execution_on_backend_with_input_shapes(
            static_profile,
            bundle,
            Backend.CPU,
            input_shapes,
        )
        assert profile.backend == Backend.CPU
        assert set(profile.layer_profiles) == {"sum", "linear"}
        assert all(
            layer.execution_time_s >= 0 for layer in profile.layer_profiles.values()
        )


def test_controller_and_worker_use_the_same_concrete_input_shapes(
    profiled_variant: ProfiledModelVariant,
) -> None:
    dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(profiled_variant)
    dto.profile.variable_properties = dto.profile.variable_properties[:1]
    worker_static = ModelStaticProfileMapper.model_static_profile_from_dto(dto)
    worker_id = ServiceId()
    input_shapes = ModelInputConcreteShapes(
        dict(
            reversed(
                tuple(worker_static.input_shape_configurations[0].input_shapes.items())
            )
        )
    )
    execution = ModelExecutionProfile(
        model_version_id=worker_static.model_variant_id,
        shape_execution_profiles=(
            ShapeExecutionProfile(
                input_shapes=input_shapes,
                backend_profiles=(
                    BackendExecutionProfile(
                        backend=Backend.CPU,
                        layer_profiles={
                            "sum": LayerExecutionProfile(0.1, 16),
                            "linear": LayerExecutionProfile(0.2, 24),
                        },
                    ),
                ),
            ),
        ),
    )
    execution_dto = ModelExecutionProfileMapper.model_execution_profile_to_dto(
        worker_id, execution
    )
    execution_dto = ModelExecutionProfileDto.model_validate_json(
        execution_dto.model_dump_json()
    )
    execution_profile = (
        ModelVariantExecutionProfileMapper.model_variant_execution_profile_from_dto(
            execution_dto
        )
    )
    static = ModelVariantStaticProfileMapper.model_variant_static_profile_from_dto(
        dto,
        ModelDto(
            model_id=dto.model_variant_id.model_id,
            model_task=ModelTask.CLASSIFICATION,
            architecture_info=CNNArchitectureInfo(),
        ),
    )
    assert (
        static.shape_profiles[0].input_shapes
        == execution_profile.shape_execution_profiles[0].input_shapes
    )
    batch = batch_size(static)
    assert tensor_size_bytes(static, "output") == batch * 2 * 4
    assert layer_weight_size_bytes(static, "linear") == 3 * 2 * 4
    context = OptimizationContext(
        static_profiles_by_variant={static.model_variant_id: static},
        network_topology=NetworkTopology(workers=[worker_id], connections=[]),
        resource_profiles_by_worker={},
        execution_profiles_by_worker={
            worker_id: WorkerExecutionProfiles(
                worker_id, {static.model_variant_id: execution_profile}
            )
        },
    )
    layers = layer_execution_profiles(context, static.model_variant_id, worker_id)
    assert layers is not None
    assert layers["linear"].execution_time_s == 0.2
    assert layers["InputLayer"].execution_time_s == 0.0
