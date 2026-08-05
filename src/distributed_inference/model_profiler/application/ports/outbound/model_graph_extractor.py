import itertools
from abc import ABC, abstractmethod

from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model_version import ModelVersionInfo
from distributed_inference.model_manager.domain.model_version_graph import (
    ModelVersionGraph,
    ShapePoint,
)


## NOTE: For now, we keep these calls synchronous since they are compute intensive
class ModelGraphExtractor(ABC):
    @abstractmethod
    def extract_model_graph(
        self,
        paths: ArtifactWorkspace,
        model_version_info: ModelVersionInfo,
        profile_flops: bool,
        profile_tensors: bool,
    ) -> ModelVersionGraph: ...

    @abstractmethod
    def aggregate_model_graphs(
        self, level_1_graph: ModelVersionGraph, level_2_graph: ModelVersionGraph
    ) -> ModelVersionGraph: ...

    @classmethod
    def compute_shape_points(
        cls, model_version_info: ModelVersionInfo
    ) -> list[ShapePoint]:
        static_shapes = model_version_info.static_shapes
        dynamic_shapes = model_version_info.dynamic_shapes

        base_shape_point: list[tuple[str, int]] = []
        for static_shape in static_shapes:
            static_shape_tuple = (
                static_shape.name,
                static_shape.value,
            )
            base_shape_point.append(static_shape_tuple)

        if not dynamic_shapes:
            return [ShapePoint(dims=tuple(sorted(base_shape_point)))]

        options_per_dynamic_shape = []
        for dynamic_shape in dynamic_shapes:
            options = []
            for value in range(
                dynamic_shape.min_value,
                dynamic_shape.max_value + 1,
                dynamic_shape.step_size,
            ):
                options.append((dynamic_shape.name, value))
            options_per_dynamic_shape.append(options)

        return [
            ShapePoint(dims=tuple(sorted(base_shape_point + list(combo))))
            for combo in itertools.product(*options_per_dynamic_shape)
        ]
