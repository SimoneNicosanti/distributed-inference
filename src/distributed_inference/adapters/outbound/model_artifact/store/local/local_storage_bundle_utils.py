import fcntl
import shutil
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
    MaterializedArtifact,
)

MANIFEST_FILE_NAME = "manifest.json"


def put_bundle(
    bundle: ArtifactBundle, bundle_root_path: Path, lock_file_path: Path
) -> None:
    with lock_file_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            for artifact_file in bundle.artifact_files:
                file_path = bundle_root_path.joinpath(*artifact_file.rel_path.parts)
                file_path.parent.mkdir(parents=True, exist_ok=True)

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

            with ExitStack() as stack:
                artifact_files: list[ArtifactFile] = []
                for rel_file_path in manifest.rel_file_paths:
                    file_path = bundle_root_path.joinpath(*rel_file_path.parts)

                    bundle_file_content = stack.enter_context(file_path.open("rb"))

                    artifact_file = ArtifactFile(
                        rel_path=rel_file_path,
                        content=bundle_file_content,
                    )
                    artifact_files.append(artifact_file)

                yield ArtifactBundle(
                    manifest=manifest,
                    artifact_files=tuple(artifact_files),
                )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def check_bundle(bundle_root_path: Path, lock_path: Path) -> bool:
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
        try:
            return bundle_root_path.exists() and manifest_path.exists()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def get_bundle_materialized_artifact(
    bundle_root_path: Path, lock_path: Path
) -> Generator[MaterializedArtifact]:
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            manifest_path = bundle_root_path.joinpath(MANIFEST_FILE_NAME)
            manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )

            materialized_artifact = MaterializedArtifact(
                root_path=bundle_root_path,
                entrypoint_path=bundle_root_path.joinpath(
                    *manifest.rel_entrypoint_path.parts
                ),
            )
            yield materialized_artifact
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
