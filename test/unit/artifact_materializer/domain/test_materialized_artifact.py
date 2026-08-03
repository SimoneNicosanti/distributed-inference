from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)


def _manifest() -> ArtifactManifest:
    entrypoint = PurePosixPath("model.onnx")
    return ArtifactManifest(
        entrypoint_ppp=entrypoint,
        files_info=(ArtifactFileInfo(file_ppp=entrypoint),),
    )


@pytest.mark.unit
def test_materialized_artifact_requires_existing_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not a directory"):
        MaterializedArtifact(
            manifest=_manifest(),
            root_path=tmp_path / "missing",
            entrypoint_path=tmp_path / "missing" / "model.onnx",
        )


@pytest.mark.unit
def test_materialized_artifact_rejects_entrypoint_outside_root(tmp_path: Path) -> None:
    root_path = tmp_path / "bundle"
    root_path.mkdir()
    outside_entrypoint = tmp_path / "outside.onnx"
    outside_entrypoint.write_bytes(b"model")

    with pytest.raises(ValidationError, match="contained inside"):
        MaterializedArtifact(
            manifest=_manifest(),
            root_path=root_path,
            entrypoint_path=outside_entrypoint,
        )
