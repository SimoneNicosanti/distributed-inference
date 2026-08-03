from io import BytesIO
from pathlib import PurePosixPath
from typing import BinaryIO

import pytest
from pydantic import ValidationError

from distributed_inference.model_artifact.adapters.outbound.local.local_artifact_bundle_builder import (
    build_artifact_bundle_from_bundle_paths,
    build_manifest_from_bundle_paths,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)
from distributed_inference.model_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)


@pytest.mark.unit
def test_build_manifest_recursively_lists_files_and_preserves_entrypoint(
    tmp_path,
) -> None:
    entrypoint = tmp_path / "model.onnx"
    weights = tmp_path / "weights" / "model.data"
    weights.parent.mkdir()
    entrypoint.write_bytes(b"model")
    weights.write_bytes(b"weights")
    (tmp_path / "empty-directory").mkdir()

    manifest: ArtifactManifest = build_manifest_from_bundle_paths(
        MaterializedArtifact(
            root_path=tmp_path,
            entrypoint_path=entrypoint,
        )
    )

    assert manifest.rel_entrypoint_path == PurePosixPath("model.onnx")
    assert set(manifest.rel_file_paths) == {
        PurePosixPath("model.onnx"),
        PurePosixPath("weights/model.data"),
    }


@pytest.mark.unit
def test_bundle_builder_keeps_all_streams_open_only_inside_context(tmp_path) -> None:
    entrypoint = tmp_path / "model.onnx"
    config = tmp_path / "config.json"
    entrypoint.write_bytes(b"onnx-bytes")
    config.write_bytes(b'{"format": "onnx"}')
    streams = []

    with build_artifact_bundle_from_bundle_paths(
        MaterializedArtifact(
            root_path=tmp_path,
            entrypoint_path=entrypoint,
        )
    ) as bundle:
        streams: list[BinaryIO] = [
            artifact.content for artifact in bundle.artifact_files
        ]
        payloads: dict[PurePosixPath, bytes] = {
            artifact.rel_path: artifact.content.read()
            for artifact in bundle.artifact_files
        }

        assert payloads == {
            PurePosixPath("model.onnx"): b"onnx-bytes",
            PurePosixPath("config.json"): b'{"format": "onnx"}',
        }
        assert all(not stream.closed for stream in streams)

    assert all(stream.closed for stream in streams)


@pytest.mark.unit
def test_manifest_requires_an_entrypoint(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_manifest_from_bundle_paths(MaterializedArtifact(root_path=tmp_path))


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/absolute.onnx"),
        PurePosixPath("../outside.onnx"),
        PurePosixPath("manifest.json"),
        PurePosixPath("nested\\model.onnx"),
    ],
)
def test_artifact_file_rejects_unsafe_or_reserved_paths(path) -> None:
    with pytest.raises(ValueError):
        ArtifactFile(rel_path=path, content=BytesIO(b"model"))


@pytest.mark.unit
def test_bundle_requires_manifest_and_files_to_describe_same_paths() -> None:
    model_path = PurePosixPath("model.onnx")
    manifest = ArtifactManifest(
        rel_entrypoint_path=model_path,
        rel_file_paths=(model_path,),
    )

    with pytest.raises(ValueError, match="do not match"):
        ArtifactBundle(
            manifest=manifest,
            artifact_files=(
                ArtifactFile(
                    rel_path=PurePosixPath("other.onnx"),
                    content=BytesIO(b"model"),
                ),
            ),
        )


@pytest.mark.unit
def test_concrete_paths_reject_entrypoint_outside_bundle_root(tmp_path) -> None:
    root_path = tmp_path / "bundle"
    root_path.mkdir()
    outside_entrypoint = tmp_path / "outside.onnx"
    outside_entrypoint.write_bytes(b"model")

    with pytest.raises(ValidationError, match="contained inside"):
        MaterializedArtifact(
            root_path=root_path,
            entrypoint_path=outside_entrypoint,
        )
