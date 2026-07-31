import asyncio
import fcntl
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from distributed_inference.model_artifact.domain import (
    artifact_bundle_builder,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    MANIFEST_FILE_NAME,
    ArtifactBundle,
    ArtifactConcretePaths,
    ArtifactManifest,
)

## TODO: Try to better handle the async management of local artifact store


async def put_bundle(
    bundle: ArtifactBundle, bundle_root_path: Path, lock_file_path: Path
) -> None:
    return await asyncio.to_thread(
        _put_bund_sync, bundle, bundle_root_path, lock_file_path
    )


def _put_bund_sync(
    bundle: ArtifactBundle, bundle_root_path: Path, lock_file_path: Path
) -> None:
    with lock_file_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        ## TODO: We should ensure consistency of the bundle for example when a bundle is changed with a new version
        try:
            for artifact_file in bundle.artifact_files:
                file_path = bundle_root_path.joinpath(*artifact_file.rel_path.parts)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                artifact_file.content.seek(0)
                with file_path.open("wb") as bundle_file:
                    shutil.copyfileobj(artifact_file.content, bundle_file)

            manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
            with manifest_path.open("w+") as manifest_file:
                manifest_file.write(bundle.manifest.model_dump_json())

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def get_bundle(bundle_root_path: Path, lock_path: Path) -> Generator[ArtifactBundle]:
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
            manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )

            with artifact_bundle_builder.build_bundle_from_root_path_and_manifest(
                bundle_root_path, manifest
            ) as artifact_bundle:
                yield artifact_bundle

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


async def check_bundle(bundle_root_path: Path, lock_path: Path) -> bool:
    return await asyncio.to_thread(_check_bundle_sync, bundle_root_path, lock_path)


def _check_bundle_sync(bundle_root_path: Path, lock_path: Path) -> bool:
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
        try:
            return bundle_root_path.exists() and manifest_path.exists()
            ## TODO We should check for the existence of the whole bundle as declared in the manifest
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def get_bundle_materialized_artifact(
    bundle_root_path: Path, lock_path: Path
) -> Generator[ArtifactConcretePaths]:
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
            manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )

            materialized_artifact = ArtifactConcretePaths(
                root_path=bundle_root_path,
                entrypoint_path=bundle_root_path.joinpath(
                    *manifest.rel_entrypoint_path.parts
                ),
            )
            yield materialized_artifact
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
