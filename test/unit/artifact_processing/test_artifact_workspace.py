from pathlib import Path, PurePosixPath

import pytest

from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
    build_local_artifact_bundle_from_artifact_workspace,
    build_local_artifact_bundle_from_root_path_and_manifest,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)


@pytest.mark.unit
def test_workspace_builder_recursively_lists_files_and_entrypoint(tmp_path: Path) -> None:
    entrypoint = tmp_path / "model.onnx"
    weights = tmp_path / "weights" / "model.data"
    weights.parent.mkdir()
    entrypoint.write_bytes(b"model")
    weights.write_bytes(b"weights")
    (tmp_path / "empty-directory").mkdir()

    bundle = build_local_artifact_bundle_from_artifact_workspace(
        ArtifactWorkspace(
            root_path=tmp_path,
            entrypoint_path=entrypoint,
        )
    )

    assert bundle.manifest.entrypoint_ppp == PurePosixPath("model.onnx")
    assert bundle.manifest.get_ppp_set() == {
        PurePosixPath("model.onnx"),
        PurePosixPath("weights/model.data"),
    }


@pytest.mark.unit
def test_workspace_builder_requires_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Entrypoint path must be set"):
        build_local_artifact_bundle_from_artifact_workspace(
            ArtifactWorkspace(root_path=tmp_path)
        )


@pytest.mark.unit
def test_root_and_manifest_builder_rejects_missing_declared_file(tmp_path: Path) -> None:
    entrypoint = PurePosixPath("model.onnx")
    manifest = ArtifactManifest(
        entrypoint_ppp=entrypoint,
        files_info=(ArtifactFileInfo(file_ppp=entrypoint),),
    )

    with pytest.raises(ValueError, match="File does not exist"):
        build_local_artifact_bundle_from_root_path_and_manifest(tmp_path, manifest)
