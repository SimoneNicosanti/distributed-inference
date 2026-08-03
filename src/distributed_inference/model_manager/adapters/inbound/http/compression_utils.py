import asyncio
import zipfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import aiofiles.os
from fastapi import UploadFile

from distributed_inference.artifact_processing.artifact_workspace import (
    build_local_artifact_bundle_from_root_path_and_manifest,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)

CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def decompress_artifact_bundle(
    bundle_file: UploadFile,
) -> AsyncGenerator[ReadableArtifactBundle]:

    await bundle_file.seek(0)

    async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
        artifact_bundle = await asyncio.to_thread(
            _readable_artifuct_bundle_build_sync, tmp_dir, bundle_file
        )
        yield artifact_bundle


def _readable_artifuct_bundle_build_sync(
    tmp_dir_str: str, bundle_file: UploadFile
) -> ReadableArtifactBundle:
    extraction_path = Path(tmp_dir_str).resolve(strict=True)
    with zipfile.ZipFile(bundle_file.file, mode="r") as zip_file:
        ## TODO We should do a lot of check regarding the security of zip extraction
        zip_file.extractall(path=extraction_path)

        manifest_path = extraction_path.joinpath(MANIFEST_FILE_NAME)
        if not manifest_path.is_file():
            raise Exception(
                "Zip file does not contain a manifest or manifest not in the root of the zip file"
            )

        zip_manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

        artifact_bundle = build_local_artifact_bundle_from_root_path_and_manifest(
            extraction_path, zip_manifest
        )
        return artifact_bundle


@asynccontextmanager
async def compress_artifact_bundle(
    bundle: ReadableArtifactBundle,
) -> AsyncGenerator[Path]:

    async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
        tmp_bundle_path = Path(tmp_dir).joinpath("artifact")
        await aiofiles.os.makedirs(tmp_bundle_path, exist_ok=True)
        ## We first copy the files to a temporary directory in order
        ## to make the zip funziont awaitable

        manifest = bundle.manifest
        for file_info in manifest.files_info:
            file_ppp = file_info.file_ppp
            async with bundle.open_file(file_ppp) as artifact_file:
                tmp_bundle_file_path = tmp_bundle_path.joinpath(*file_ppp.parts)
                await aiofiles.os.makedirs(tmp_bundle_file_path.parent, exist_ok=True)

                async with aiofiles.open(
                    tmp_bundle_file_path,
                    mode="wb",
                ) as tmp_bundle_file:
                    while chunk := await artifact_file.read(CHUNK_SIZE):
                        await tmp_bundle_file.write(chunk)

        tmp_manifest_file_path = tmp_bundle_path.joinpath(MANIFEST_FILE_NAME)
        async with aiofiles.open(tmp_manifest_file_path, "w+") as manifest_file:
            await manifest_file.write(bundle.manifest.model_dump_json())

        zip_file_path = Path(tmp_dir).joinpath("artifact.zip")
        await asyncio.to_thread(_zip_build_sync, tmp_bundle_path, zip_file_path)

        ## We return the path to the zip file; we do not return the zip
        ## file instance itself not to bind the compression format
        yield zip_file_path


def _zip_build_sync(tmp_bundle_path: Path, zip_file_path: Path) -> None:

    with zipfile.ZipFile(
        zip_file_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zip_file:
        for bundle_file_path in tmp_bundle_path.rglob("*"):
            if bundle_file_path.is_file():
                zip_file.write(
                    bundle_file_path,
                    arcname=bundle_file_path.relative_to(tmp_bundle_path).as_posix(),
                )
