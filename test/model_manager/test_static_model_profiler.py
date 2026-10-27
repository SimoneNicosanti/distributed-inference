import asyncio
from pathlib import Path

import onnx
from onnx_tool.node import ReshapeNode

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.adapters.outbound.model_optimizer.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from model_manager.adapters.outbound.model_profiler.onnx_contraction_computer import (
    OnnxContractionComputer,
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
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)
from shared.model.model import CNNArchitectureInfo, ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)

original_shape_infer = ReshapeNode.shape_infer


def debug_shape_infer(self, intensors, outtensors):
    try:
        return original_shape_infer(self, intensors, outtensors)
    except AssertionError:
        print("Failing Reshape:", self.name)
        print("Source shape:", intensors[0].get_shape())
        print("Target shape:", intensors[1].get_numpy())
        print("Input tensors:", self.input)
        raise


ReshapeNode.shape_infer = debug_shape_infer

# import numpy as np
# from onnx_tool.node import ShapeNode


# def corrected_shape_value_infer(self, intensors, outtensors):
#     shape = np.asarray(
#         intensors[0].get_shape(),
#         dtype=np.int64,
#     )

#     start = getattr(self, "start", None)
#     end = getattr(self, "end", None)

#     result = shape[start:end]

#     outtensors[0].update_tensor(result)


# ShapeNode.value_infer = corrected_shape_value_infer


async def main() -> None:
    # Adjust relative to the working directory
    model_path = Path("./assets/models/resnet50_optimized.onnx")

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    # Read the actual input dimensions from ONNX
    onnx_model = onnx.load(str(model_path))

    input_tensor = next(
        value for value in onnx_model.graph.input if value.name == "images"
    )

    dimensions = input_tensor.type.tensor_type.shape.dim

    input_shape = tuple(
        dim.dim_param if dim.dim_param else dim.dim_value for dim in dimensions
    )

    assert len(input_shape) == 4
    assert input_shape[1:] == (3, 224, 224)

    batch_dimension = input_shape[0]

    if not isinstance(batch_dimension, str) or not batch_dimension:
        raise ValueError("Expected a dynamic batch dimension")

    print(f"Input shape: {input_shape}")

    model_profiler = DefaultModelProfiler(
        OnnxModelTopologyBuilder(),
        OnnxInvariantPropertiesComputer(),
        OnnxShapeProfileComputer(),
        OnnxContractionComputer(
            OnnxModelOptimizer(),
            OnnxModelTopologyBuilder(),
        ),
    )

    artifact_workspace = ArtifactWorkspace(
        model_path.parent,
        model_path,
    )

    print(artifact_workspace)

    model_info = ModelInfo(
        model_task=ModelTask.CLASSIFICATION,
        architecture_info=CNNArchitectureInfo(),
    )

    model_variant_info = ModelVariantInfo(
        precision=NumericPrecision.FP32,
        quantization=QuantizationType.NONE,
        accuracies=[],
        format=ModelVariantFormat.ONNX,
        inputs_info={
            "images": InputInfo(shape=input_shape),
        },
        dynamic_dimension_values={
            batch_dimension: DynamicShapeValues(
                values=frozenset({1, 2, 4, 8}),
            ),
        },
    )

    model_variant_id = ModelVariantId(
        model_id=ModelId(
            model_name="resnet50",
        ),
        variant_tag="nano",
    )

    model_variant = ModelVariant(
        model_variant_id=model_variant_id,
        model_variant_info=model_variant_info,
    )

    profile = await model_profiler.profile_model_variant(
        artifact_workspace=artifact_workspace,
        model_info=model_info,
        model_variant=model_variant,
    )

    print(profile)


if __name__ == "__main__":
    asyncio.run(main())
