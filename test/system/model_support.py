import zipfile
from pathlib import Path, PurePosixPath

import onnx

from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from integration.dto.model.model_variant_dto import DynamicShapeValuesDto, InputInfoDto


def _read_model_inputs(
    model: onnx.ModelProto,
) -> tuple[dict[str, InputInfoDto], dict[str, DynamicShapeValuesDto]]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs_info: dict[str, InputInfoDto] = {}
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto] = {}

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        dimensions: list[str | int] = []
        for index, dimension in enumerate(value_info.type.tensor_type.shape.dim):
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                dimensions.append(dimension.dim_value)
                continue

            name = dimension.dim_param or f"{value_info.name}_dim_{index}"
            dimensions.append(name)
            dynamic_dimension_values.setdefault(
                name,
                DynamicShapeValuesDto(values=frozenset({1, 2, 4, 8})),
            )

        inputs_info[value_info.name] = InputInfoDto(shape=tuple(dimensions))

    if not inputs_info:
        raise ValueError("ONNX model has no runtime inputs")

    print(inputs_info)
    print(dynamic_dimension_values)

    return inputs_info, dynamic_dimension_values


def _build_model_archive(model_path: Path, archive_path: Path) -> None:
    artifact_model_path = PurePosixPath(model_path.name)
    manifest = ArtifactManifest(
        entrypoint_ppp=artifact_model_path,
        files_info=(ArtifactFileInfo(file_ppp=artifact_model_path),),
    )
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as file:
        file.write(model_path, arcname=artifact_model_path.as_posix())
        file.writestr("manifest.json", manifest.model_dump_json())
