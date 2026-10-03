import asyncio
import zipfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles

from artifacts.contracts.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from artifacts.contracts.artifact_bundle import ArtifactBundle


@asynccontextmanager
async def decompress_artifact_bundle(
    zip_file_path: Path,
) -> AsyncGenerator[ArtifactBundle]:

    async with aiofiles.tempfile.TemporaryDirectory() as extraction_dir:
        extraction_dir_path = Path(extraction_dir)
        artifact_source = await asyncio.to_thread(
            _artifact_source_build_sync, extraction_dir_path, zip_file_path
        )
        yield artifact_source


def _artifact_source_build_sync(
    extraction_dir_path: Path, zip_file_path: Path
) -> ArtifactBundle:
    with zipfile.ZipFile(zip_file_path, mode="r") as zip_file:
        ## TODO We should do a lot of check regarding the security of zip extraction
        zip_file.extractall(path=extraction_dir_path)

        manifest_path = extraction_dir_path.joinpath(MANIFEST_FILE_NAME)
        if not manifest_path.is_file():
            raise Exception(
                "Zip file does not contain a manifest or manifest not in the root of the zip file"
            )

        zip_manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

        return ArtifactBundle(manifest=zip_manifest, root_path=extraction_dir_path)


@asynccontextmanager
async def compress_artifact_bundle(
    source: ArtifactBundle,
) -> AsyncGenerator[Path]:

    async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
        zip_file_path = Path(tmp_dir).joinpath("artifact.zip")
        await asyncio.to_thread(_zip_build_sync, source, zip_file_path)

        yield zip_file_path


def _zip_build_sync(source: ArtifactBundle, zip_file_path: Path) -> None:
    with zipfile.ZipFile(
        zip_file_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zip_file:
        for file_info in source.manifest.files_info:
            zip_file.write(
                source.root_path.joinpath(*file_info.file_ppp.parts),
                arcname=file_info.file_ppp.as_posix(),
            )

        zip_file.writestr(
            MANIFEST_FILE_NAME,
            source.manifest.model_dump_json(),
        )
