from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)

from collections.abc import Generator
from typing import BinaryIO, override, Tuple

from distributed_inference.domain.identifiers import (
    SubModelId,
)

from distributed_inference.domain.model_graph_info import LayerKey

from contextlib import contextmanager

import shutil

from pathlib import Path

import fcntl


class LocalSubModelArtifactStore(SubModelArtifactStore):
    def __init__(
        self,
        base_path: Path,
    ):
        self.base_path = base_path

        self.sub_models_dir = base_path / "sub_models"
        self.lock_dir = self.sub_models_dir / ".lock"

        base_path.mkdir(parents=True, exist_ok=True)
        self.sub_models_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @override
    def put_sub_model(
        self,
        sub_model_id: SubModelId,
        binary_io: BinaryIO,
    ):
        file_path, lock_path = self._build_sub_model_file_path(sub_model_id)

        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with file_path.open("wb") as sub_model_file:
                    shutil.copyfileobj(binary_io, sub_model_file)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @override
    @contextmanager
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> Generator[BinaryIO]:

        file_path, lock_path = self._build_sub_model_file_path(sub_model_id)

        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                with file_path.open("rb") as model_version_file:
                    yield model_version_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def get_sub_model_path(self, sub_model_id: SubModelId) -> Generator[Path]:
        file_path, lock_path = self._build_sub_model_file_path(sub_model_id)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                yield file_path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _build_sub_model_file_path(self, sub_model_id: SubModelId) -> Tuple[Path, Path]:

        model_id = sub_model_id.model_version_id.model_id

        layers_hash = self._hash_layers(sub_model_id.layers)

        file_path = (
            self.sub_models_dir
            / str(model_id.user_id)
            / model_id.model_name
            / f"{sub_model_id.model_version_id.version_number}"
            / f"layers_{layers_hash}.onnx"
        )

        lock_path = (
            self.lock_dir
            / f"{model_id.user_id}_{model_id.model_name}_{sub_model_id.model_version_id.version_number}_{layers_hash}.lock"
        )

        return file_path, lock_path

    def _hash_layers(self, layers: Tuple[LayerKey, ...]) -> str:
        from hashlib import md5

        payload = "\0".join(str(layer) for layer in layers)
        return md5(payload.encode("utf-8")).hexdigest()

    @override
    def check_sub_model_existance(self, sub_model_id: SubModelId) -> bool:
        file_path, lock_path = self._build_sub_model_file_path(sub_model_id)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return file_path.exists()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        pass
