from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)

from collections.abc import Generator
from typing import BinaryIO, override, Tuple


from distributed_inference.domain.identifiers import (
    ModelVersionId,
)


from contextlib import contextmanager

import shutil


from pathlib import Path


import fcntl


class LocalModelVersionArtifactStore(ModelVersionArtifactStore):
    def __init__(self, base_path: Path):
        self.base_path = base_path

        self.model_versions_dir = base_path / "model_versions"
        self.lock_dir = self.model_versions_dir / ".lock"

        base_path.mkdir(parents=True, exist_ok=True)
        self.model_versions_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @override
    def put_model_version(
        self,
        model_version_id: ModelVersionId,
        binary_io: BinaryIO,
    ) -> None:
        file_path, lock_file_path = self._build_model_version_file_path(
            model_version_id
        )

        with lock_file_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with file_path.open("wb") as model_version_file:
                    shutil.copyfileobj(binary_io, model_version_file)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        pass

    @override
    @contextmanager
    def get_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> Generator[BinaryIO]:

        file_path, lock_path = self._build_model_version_file_path(model_version_id)

        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                with file_path.open("rb") as model_version_file:
                    yield model_version_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @override
    def check_model_version_existance(
        self,
        model_version_id: ModelVersionId,
    ):
        file_path, lock_path = self._build_model_version_file_path(model_version_id)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return file_path.exists()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        pass

    @contextmanager
    def get_model_version_path(
        self, model_version_id: ModelVersionId
    ) -> Generator[Path]:
        file_path, lock_path = self._build_model_version_file_path(model_version_id)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                yield file_path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _build_model_version_file_path(
        self, model_version_id: ModelVersionId
    ) -> Tuple[Path, Path]:
        model_id = model_version_id.model_id
        file_path = (
            self.model_versions_dir
            / str(model_id.user_id)
            / model_id.model_name
            / f"version_{model_version_id.version_number}.onnx"
        )
        lock_path = (
            self.lock_dir
            / f"{model_id.user_id}_{model_id.model_name}_{model_version_id.version_number}.lock"
        )
        return file_path, lock_path
