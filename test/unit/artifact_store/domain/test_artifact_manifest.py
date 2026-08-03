from pathlib import PurePosixPath

import pytest
from pydantic import ValidationError

from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/absolute.onnx"),
        PurePosixPath("../outside.onnx"),
        PurePosixPath("manifest.json"),
        PurePosixPath("nested\\model.onnx"),
        PurePosixPath("."),
    ],
)
def test_artifact_file_info_rejects_unsafe_or_reserved_paths(
    path: PurePosixPath,
) -> None:
    with pytest.raises(ValidationError):
        ArtifactFileInfo(file_ppp=path)


@pytest.mark.unit
def test_manifest_requires_declared_entrypoint() -> None:
    with pytest.raises(ValidationError, match="entrypoint is not declared"):
        ArtifactManifest(
            entrypoint_ppp=PurePosixPath("model.onnx"),
            files_info=(
                ArtifactFileInfo(file_ppp=PurePosixPath("weights.data")),
            ),
        )


@pytest.mark.unit
def test_manifest_rejects_duplicate_paths() -> None:
    file_info = ArtifactFileInfo(file_ppp=PurePosixPath("model.onnx"))

    with pytest.raises(ValidationError, match="duplicate paths"):
        ArtifactManifest(
            entrypoint_ppp=file_info.file_ppp,
            files_info=(file_info, file_info),
        )
