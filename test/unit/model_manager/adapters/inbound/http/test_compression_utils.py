from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import cast
from zipfile import ZipFile

import pytest
from fastapi import UploadFile

from distributed_inference.artifact_store.adapters.outbound.local.local_readable_artifact_bundle import (
    LocalReadableArtifactBundle,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactFileInfo,
    ArtifactManifest,
)
from distributed_inference.artifact_processing.compression_utils import (
    compress_artifact_bundle,
    decompress_artifact_bundle,
)
from test.support.artifact_store.artifact_bundle_test_utils import (
    build_test_bundle,
    read_bundle_content,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compress_bundle_writes_manifest_and_artifact_files(
    tmp_path: Path,
) -> None:
    bundle = build_test_bundle(
        tmp_path / "input",
        files={
            PurePosixPath("model/model.onnx"): b"onnx-content",
            PurePosixPath("config.json"): b'{"version": 1}',
        },
        entrypoint=PurePosixPath("model/model.onnx"),
    )

    async with compress_artifact_bundle(bundle) as zip_path:
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
@pytest.mark.asyncio
async def test_decompress_bundle_exposes_nested_files_for_context_lifetime() -> None:
    archive_bytes = BytesIO()
    manifest = ArtifactManifest(
        entrypoint_ppp=PurePosixPath("model/model.onnx"),
        files_info=(
            ArtifactFileInfo(file_ppp=PurePosixPath("model/model.onnx")),
            ArtifactFileInfo(file_ppp=PurePosixPath("labels.txt")),
        ),
    )
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr(MANIFEST_FILE_NAME, manifest.model_dump_json())
        archive.writestr("model/model.onnx", b"onnx-content")
        archive.writestr("labels.txt", b"cat\ndog\n")
    archive_bytes.seek(0)
    upload = UploadFile(filename="artifact.zip", file=archive_bytes)

    async with decompress_artifact_bundle(upload) as readable:
        bundle = cast(LocalReadableArtifactBundle, readable)
        extracted_root = bundle.local_root_path
        assert bundle.manifest == manifest
        assert await read_bundle_content(bundle) == {
            PurePosixPath("model/model.onnx"): b"onnx-content",
            PurePosixPath("labels.txt"): b"cat\ndog\n",
        }
        assert extracted_root.is_dir()

    assert not extracted_root.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decompress_bundle_requires_root_manifest() -> None:
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr("model.onnx", b"model")
    archive_bytes.seek(0)
    upload = UploadFile(filename="artifact.zip", file=archive_bytes)

    with pytest.raises(Exception, match="does not contain a manifest"):
        async with decompress_artifact_bundle(upload):
            pass
