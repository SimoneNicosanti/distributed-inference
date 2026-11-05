from typing import Any

import pytest
from pydantic import ValidationError

from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile as ControllerExecution,
)
from controller.domain.profile.model_variant_static_profile import (
    LayerShapeProperty,
    TensorProperty,
    TensorShapeProperty,
    WeightProperty,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelInputConcreteShapes as ControllerInputs,
)
from integration.dto.model.model_variant_dto import DynamicShapeValuesDto, InputInfoDto
from integration.dto.model.profiled_model_variant_dto import (
    InputShapeInvariableTensorPropertiesDto,
    InputShapeVariableLayerPropertiesDto,
    InputShapeVariableModelPropertiesDto,
    InputShapeVariableTensorPropertiesDto,
    ModelInputConcreteShapesDto,
)
from integration.dto.worker_profiles.execution_profile_dto import (
    LayerExecutionProfileDto,
)
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableLayerProperties,
    InputShapeVariableModelProperties,
    InputShapeVariableTensorProperties,
    ModelInputConcreteShapes,
)
from model_manager.domain.model_variant import DynamicShapeValues, InputInfo
from worker.domain.profiling.model_execution.model_execution_profile import (
    LayerExecutionProfile as WorkerExecution,
)
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes as WorkerInputs,
)


@pytest.mark.parametrize(
    "dimension", [0, -1, True, False, 1.0, 1.5, float("nan"), float("inf"), "1", None]
)
def test_concrete_dimensions_are_strict_positive_integers(dimension: Any) -> None:
    shape = (dimension,)
    factories = (
        lambda: ModelInputConcreteShapes({"input": shape}),
        lambda: ControllerInputs({"input": shape}),
        lambda: WorkerInputs({"input": shape}),
        lambda: InputShapeVariableTensorProperties(shape),
        lambda: TensorShapeProperty(shape),
        lambda: WeightProperty("float32", shape, 4),
        lambda: ModelInputConcreteShapesDto(input_shapes={"input": shape}),
        lambda: InputShapeVariableTensorPropertiesDto(shape=shape),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.parametrize(
    "dimension", [0, -1, True, False, 1.0, 1.5, float("nan"), float("inf"), "", None]
)
def test_symbolic_shapes_accept_only_positive_integers_or_nonempty_symbols(
    dimension: Any,
) -> None:
    shape = (dimension,)
    factories = (
        lambda: InputInfo(shape),
        lambda: InputShapeInvariableTensorProperties("float32", 4, shape),
        lambda: TensorProperty("float32", 4, shape),
        lambda: InputInfoDto(shape=shape),
        lambda: InputShapeInvariableTensorPropertiesDto(
            data_type="float32", data_type_size_in_bytes=4, shape_expression=shape
        ),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.parametrize("shape", [(), (1,), (1, 3)])
def test_scalars_and_positive_concrete_shapes_remain_valid(
    shape: tuple[int, ...],
) -> None:
    assert ModelInputConcreteShapes({"input": shape}).input_shapes["input"] == shape
    assert ControllerInputs({"input": shape}).input_shapes["input"] == shape
    assert WorkerInputs({"input": shape}).input_shapes["input"] == shape
    dto = ModelInputConcreteShapesDto(input_shapes={"input": shape})
    assert (
        ModelInputConcreteShapesDto.model_validate_json(
            dto.model_dump_json()
        ).input_shapes["input"]
        == shape
    )
    assert InputShapeVariableTensorProperties(shape).shape == shape
    assert (
        InputShapeInvariableTensorProperties("float32", 4, shape).shape_expression
        == shape
    )


def test_valid_symbolic_shapes_and_dynamic_values_remain_valid() -> None:
    shape = ("batch", 3)
    assert InputInfo(shape).shape == shape
    assert InputInfoDto.model_validate_json('{"shape": ["batch", 3]}').shape == shape
    assert (
        InputShapeInvariableTensorProperties("float32", 4, shape).shape_expression
        == shape
    )
    assert DynamicShapeValues(frozenset({1, 2})).values == frozenset({1, 2})
    assert DynamicShapeValuesDto.model_validate_json(
        '{"values": [1, 2]}'
    ).values == frozenset({1, 2})


@pytest.mark.parametrize(
    "values",
    [
        frozenset(),
        frozenset({0}),
        frozenset({-1}),
        frozenset({True}),
        frozenset({1.0}),
        frozenset({"1"}),
    ],
)
def test_dynamic_shape_samples_are_nonempty_and_strict(values: Any) -> None:
    with pytest.raises(ValueError):
        DynamicShapeValues(values)
    with pytest.raises(ValidationError):
        DynamicShapeValuesDto(values=values)


@pytest.mark.parametrize(
    "size", [0, -1, True, False, 1.0, 1.5, float("nan"), float("inf"), "4", None]
)
def test_element_size_is_a_positive_integer_byte_count(size: Any) -> None:
    factories = (
        lambda: InputShapeInvariableTensorProperties("float32", size, (1,)),
        lambda: TensorProperty("float32", size, (1,)),
        lambda: WeightProperty("float32", (1,), size),
        lambda: InputShapeInvariableTensorPropertiesDto(
            data_type="float32", data_type_size_in_bytes=size, shape_expression=(1,)
        ),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.parametrize(
    "value", [-1, True, False, float("nan"), float("inf"), float("-inf"), "1", None]
)
def test_numeric_profile_metrics_reject_invalid_values(value: Any) -> None:
    factories = (
        lambda: InputShapeVariableLayerProperties(value),
        lambda: LayerShapeProperty(value),
        lambda: ControllerExecution(value),
        lambda: WorkerExecution(value, None),
        lambda: InputShapeVariableLayerPropertiesDto(flops=value),
        lambda: LayerExecutionProfileDto(execution_time_s=value, memory_bytes=None),
        lambda: InputShapeVariableModelProperties(value),
        lambda: InputShapeVariableModelPropertiesDto(max_cut_size_mib=value),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.parametrize(
    "memory", [-1, True, False, 1.0, 1.5, float("nan"), float("inf"), "1"]
)
def test_execution_memory_is_an_integer_byte_count(memory: Any) -> None:
    factories = (
        lambda: ControllerExecution(0.0, memory),
        lambda: WorkerExecution(0.0, memory),
        lambda: LayerExecutionProfileDto(execution_time_s=0.0, memory_bytes=memory),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


def test_zero_costs_and_unknown_memory_remain_valid() -> None:
    assert InputShapeVariableLayerProperties(0).flops == 0
    assert LayerShapeProperty(0).flops == 0
    assert InputShapeVariableLayerPropertiesDto(flops=0).flops == 0
    assert ControllerExecution(0, 0).memory_bytes == 0
    assert WorkerExecution(0, None).memory_bytes is None
    assert (
        LayerExecutionProfileDto(execution_time_s=0, memory_bytes=0).memory_bytes == 0
    )


def test_max_cut_is_positive_finite_mib_and_can_be_fractional() -> None:
    for factory in (
        lambda: InputShapeVariableModelProperties(0),
        lambda: InputShapeVariableModelPropertiesDto(max_cut_size_mib=0),
    ):
        with pytest.raises(ValueError):
            factory()
    value = 1 / (1024 * 1024)
    assert InputShapeVariableModelProperties(value).max_cut_size_mib == value
    dto = InputShapeVariableModelPropertiesDto(max_cut_size_mib=value)
    assert (
        InputShapeVariableModelPropertiesDto.model_validate_json(
            dto.model_dump_json()
        ).max_cut_size_mib
        == value
    )
