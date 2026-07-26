from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest
from fastapi import UploadFile

from distributed_inference.adapters.inbound.model_manager.http.compression_utils import (
    compress_artifact_bundle,
    decompress_artifact_bundle,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MANIFEST_FILE_NAME,
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)


def _bundle() -> ArtifactBundle:
    return ArtifactBundle(
        manifest=ArtifactManifest(
            rel_entrypoint_path=PurePosixPath("model/model.onnx"),
            rel_file_paths=(
                PurePosixPath("model/model.onnx"),
                PurePosixPath("config.json"),
            ),
        ),
        artifact_files=(
            ArtifactFile(
                rel_path=PurePosixPath("model/model.onnx"),
                content=BytesIO(b"onnx-content"),
            ),
            ArtifactFile(
                rel_path=PurePosixPath("config.json"),
                content=BytesIO(b'{"version": 1}'),
            ),
        ),
    )


@pytest.mark.unit
def test_compress_bundle_writes_manifest_and_artifact_files() -> None:
    bundle = _bundle()
    for artifact in bundle.artifact_files:
        artifact.content.seek(3)

    with compress_artifact_bundle(bundle) as zip_path:
        assert zip_path.is_file()
        with ZipFile(zip_path) as archive:
            assert set(archive.namelist()) == {
                MANIFEST_FILE_NAME,
                "model/model.onnx",
                "config.json",
            }
            assert archive.read("model/model.onnx") == b"onnx-content"
            assert archive.read("config.json") == b'{"version": 1}'
            manifest = ArtifactManifest.model_validate_json(
                archive.read(MANIFEST_FILE_NAME)
            )
            assert manifest == bundle.manifest

    assert not zip_path.exists()


@pytest.mark.unit
def test_uncompress_bundle_exposes_nested_files_for_context_lifetime() -> None:
    archive_bytes = BytesIO()
    manifest = ArtifactManifest(
        rel_entrypoint_path=PurePosixPath("model/model.onnx"),
        rel_file_paths=(PurePosixPath("model/model.onnx"), PurePosixPath("labels.txt")),
    )
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr(MANIFEST_FILE_NAME, manifest.model_dump_json())
        archive.writestr("model/model.onnx", b"onnx-content")
        archive.writestr("labels.txt", b"cat\ndog\n")
    archive_bytes.seek(0)
    upload = UploadFile(filename="artifact.zip", file=archive_bytes)
    extracted_root = None
    streams = ()

    with decompress_artifact_bundle(upload) as bundle:
        streams = tuple(artifact.content for artifact in bundle.artifact_files)
        extracted_root = Path(streams[0].name).parent.parent
        assert bundle.manifest == manifest
        assert {
            artifact.rel_path: artifact.content.read()
            for artifact in bundle.artifact_files
        } == {
            PurePosixPath("model/model.onnx"): b"onnx-content",
            PurePosixPath("labels.txt"): b"cat\ndog\n",
        }
        assert all(not stream.closed for stream in streams)

    assert extracted_root is not None
    assert all(stream.closed for stream in streams)
    assert not extracted_root.exists()


@pytest.mark.unit
def test_uncompress_bundle_requires_root_manifest() -> None:
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr("model.onnx", b"model")
    archive_bytes.seek(0)
    upload = UploadFile(filename="artifact.zip", file=archive_bytes)

    with pytest.raises(Exception, match="does not contain a manifest"):
        with decompress_artifact_bundle(upload):
            pass
