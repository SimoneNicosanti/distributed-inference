import tempfile
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from fastapi import UploadFile

from distributed_inference.application.model_artifact.domain import (
    artifact_bundle_builder,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactConcretePaths,
)

CHUNK_SIZE = 1024 * 1024


@contextmanager
def uncompress_artifact_bundle(
    bundle_file: UploadFile, entrypoint: str
) -> Generator[ArtifactBundle]:
    entrypoint_rel_path = PurePosixPath(entrypoint)

    with tempfile.TemporaryDirectory() as tmp_dir:
        extraction_path = Path(tmp_dir)
        with zipfile.ZipFile(bundle_file.file, "r") as zip_ref:
            zip_ref.extractall(path=extraction_path)

            bundle_paths = ArtifactConcretePaths(
                root_path=extraction_path,
                entrypoint_path=extraction_path.joinpath(entrypoint_rel_path),
            )

            with artifact_bundle_builder.build_artifact_bundle_from_bundle_paths(
                bundle_paths
            ) as bundle:
                yield bundle


@contextmanager
def compress_artifact_bundle(
    bundle: ArtifactBundle,
) -> Generator[Path]:

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file_path = Path(tmp_dir).joinpath("artifact.zip")

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            ## Writing the bundle manifest
            zip_file.writestr(
                ArtifactBundle.MANIFEST_FILE_NAME,
                data=bundle.manifest.model_dump_json(),
            )

            ## Writing file by file
            for artifact_file in bundle.artifact_files:
                artifact_file.content.seek(0)

                while chunk := artifact_file.content.read(CHUNK_SIZE):
                    zip_file.writestr(
                        str(artifact_file.rel_path),
                        chunk,
                    )

        ## We return the path to the zip file; we do not return the zip
        ## file instance itself not to bind the compression format
        yield zip_file_path
