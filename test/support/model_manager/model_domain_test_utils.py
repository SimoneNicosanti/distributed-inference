from uuid import uuid4

from distributed_inference.domain.identifiers import UserId
from distributed_inference.model_manager.domain.model import (
    Model,
    ModelId,
    ModelInfo,
    ModelTask,
    ModelType,
    ModelVisibility,
)
from distributed_inference.model_manager.domain.model_version import (
    AccuracyMetric,
    ArchitectureInfo,
    CNNArchitectureInfo,
    DynamicShapeInfo,
    ModelVersion,
    ModelVersionFormat,
    ModelVersionId,
    ModelVersionInfo,
    ModelVersionPrecision,
    ModelVersionQuantization,
    ProfiledModelVersion,
    ShapeType,
    StaticShapeInfo,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    LayerKey,
    ModelVersionGraph,
)
from distributed_inference.model_manager.domain.sub_model import SubModel, SubModelId


def build_model_id(
    *,
    owner_id: UserId | None = None,
    model_name: str = "resnet50",
) -> ModelId:
    return ModelId(
        owner_id=owner_id if owner_id is not None else UserId(id=uuid4()),
        model_name=model_name,
    )


def build_model_info(
    *,
    model_task: ModelTask = ModelTask.CLASSIFICATION,
    model_type: ModelType = ModelType.CNN,
) -> ModelInfo:
    return ModelInfo(model_task=model_task, model_type=model_type)


def build_model(
    *,
    model_id: ModelId | None = None,
    visibility: ModelVisibility = ModelVisibility.PRIVATE,
    model_info: ModelInfo | None = None,
) -> Model:
    return Model(
        model_id=model_id if model_id is not None else build_model_id(),
        visibility=visibility,
        model_info=model_info if model_info is not None else build_model_info(),
    )


def build_model_version_id(
    *,
    model_id: ModelId | None = None,
    version_tag: str = "v1",
) -> ModelVersionId:
    return ModelVersionId(
        model_id=model_id if model_id is not None else build_model_id(),
        version_tag=version_tag,
    )


def build_model_version_info(
    *,
    precision: ModelVersionPrecision = ModelVersionPrecision.FP32,
    quantization: ModelVersionQuantization = ModelVersionQuantization.NONE,
    accuracy: dict[AccuracyMetric, float] | None = None,
    format: ModelVersionFormat = ModelVersionFormat.ONNX,
    static_shapes: list[StaticShapeInfo] | None = None,
    dynamic_shapes: list[DynamicShapeInfo] | None = None,
    architecture_info: ArchitectureInfo | None = None,
) -> ModelVersionInfo:
    return ModelVersionInfo(
        precision=precision,
        quantization=quantization,
        accuracy=accuracy if accuracy is not None else {AccuracyMetric.MAE: 0.1},
        format=format,
        static_shapes=(
            static_shapes
            if static_shapes is not None
            else [StaticShapeInfo(type=ShapeType.BATCH, name="batch_size", value=1)]
        ),
        dynamic_shapes=dynamic_shapes if dynamic_shapes is not None else [],
        architecture_info=(
            architecture_info
            if architecture_info is not None
            else CNNArchitectureInfo()
        ),
    )


def build_model_version(
    *,
    model_version_id: ModelVersionId | None = None,
    model_version_info: ModelVersionInfo | None = None,
) -> ModelVersion:
    return ModelVersion(
        model_version_id=(
            model_version_id
            if model_version_id is not None
            else build_model_version_id()
        ),
        model_version_info=(
            model_version_info
            if model_version_info is not None
            else build_model_version_info()
        ),
    )


def build_profiled_model_version(
    *,
    model_version: ModelVersion | None = None,
    model_version_graph: ModelVersionGraph | None = None,
) -> ProfiledModelVersion:
    version = model_version if model_version is not None else build_model_version()
    return ProfiledModelVersion(
        model_version_id=version.model_version_id,
        model_version_info=version.model_version_info,
        model_version_graph=(
            model_version_graph
            if model_version_graph is not None
            else ModelVersionGraph(shape_points=[])
        ),
    )


def build_sub_model_id(
    *,
    model_version_id: ModelVersionId | None = None,
    layers: tuple[LayerKey, ...] = ("encoder.0", "encoder.1"),
) -> SubModelId:
    return SubModelId(
        model_version_id=(
            model_version_id
            if model_version_id is not None
            else build_model_version_id()
        ),
        layers=layers,
    )


def build_sub_model(*, sub_model_id: SubModelId | None = None) -> SubModel:
    return SubModel(
        sub_model_id=sub_model_id if sub_model_id is not None else build_sub_model_id()
    )
