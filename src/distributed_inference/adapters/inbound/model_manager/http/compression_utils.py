import shutil
import tempfile
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fastapi import UploadFile

from distributed_inference.application.model_artifact.domain import (
    artifact_bundle_builder,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MANIFEST_FILE_NAME,
    ArtifactBundle,
    ArtifactManifest,
)

CHUNK_SIZE = 1024 * 1024


@contextmanager
def decompress_artifact_bundle(
    bundle_file: UploadFile,
) -> Generator[ArtifactBundle]:

    bundle_file.file.seek(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        extraction_path = Path(tmp_dir).resolve(strict=True)
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

            with artifact_bundle_builder.build_bundle_from_root_path_and_manifest(
                extraction_path, zip_manifest
            ) as artifact_bundle:
                yield artifact_bundle


@contextmanager
def compress_artifact_bundle(
    bundle: ArtifactBundle,
) -> Generator[Path]:

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file_path = Path(tmp_dir).joinpath("artifact.zip")

        with zipfile.ZipFile(
            zip_file_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zip_file:
            ## Writing the bundle manifest
            zip_file.writestr(
                MANIFEST_FILE_NAME,
                data=bundle.manifest.model_dump_json(),
            )

            ## Writing file by file
            for artifact_file in bundle.artifact_files:
                artifact_file.content.seek(0)

                with zip_file.open(
                    artifact_file.rel_path.as_posix(),
                    mode="w",
                    force_zip64=True,
                ) as zip_dest_file:
                    shutil.copyfileobj(
                        artifact_file.content, zip_dest_file, length=CHUNK_SIZE
                    )

        ## We return the path to the zip file; we do not return the zip
        ## file instance itself not to bind the compression format
        yield zip_file_path
