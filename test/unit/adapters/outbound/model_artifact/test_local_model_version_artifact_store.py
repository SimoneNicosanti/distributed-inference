from io import BytesIO
from pathlib import Path
from typing import override
from uuid import uuid4

import pytest

from distributed_inference.adapters.outbound.model_artifact.store.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from test.contracts.model_version_artifact_store_contract import (
    ModelVersionArtifactStoreContract,
)


class TestLocalModelVersionArtifactStoreContract(ModelVersionArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> ModelVersionArtifactStore:
        return LocalModelVersionArtifactStore(base_path)


@pytest.fixture
def store(
    tmp_path: Path,
) -> LocalModelVersionArtifactStore:
    return LocalModelVersionArtifactStore(tmp_path)


@pytest.fixture
def model_version_id() -> ModelVersionId:
    owner_id = UserId(user_id=uuid4())
    model_id = ModelId(
        user_id=owner_id,
        model_name="resnet50",
    )

    return ModelVersionId(
        model_id=model_id,
        version_number=3,
    )


def test_constructor_creates_required_directories(
    tmp_path: Path,
) -> None:
    store = LocalModelVersionArtifactStore(tmp_path)

    assert store.base_path == tmp_path
    assert store.model_versions_dir.is_dir()
    assert store.lock_dir.is_dir()


def test_build_model_version_file_path_uses_expected_layout(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    file_path, lock_path = store._build_model_version_file_path(model_version_id)

    model_id = model_version_id.model_id

    assert file_path == (
        store.model_versions_dir
        / str(model_id.user_id)
        / model_id.model_name
        / "version_3.onnx"
    )

    assert lock_path == (
        store.lock_dir / (f"{model_id.user_id}_{model_id.model_name}_3.lock")
    )


def test_put_creates_parent_directories_and_artifact_file(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    file_path, _ = store._build_model_version_file_path(model_version_id)

    assert not file_path.parent.exists()

    store.put_model_version(
        model_version_id,
        BytesIO(b"content"),
    )

    assert file_path.is_file()
    assert file_path.read_bytes() == b"content"


def test_put_creates_lock_file(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    _, lock_path = store._build_model_version_file_path(model_version_id)

    assert not lock_path.exists()

    store.put_model_version(
        model_version_id,
        BytesIO(b"content"),
    )

    assert lock_path.is_file()


def test_put_overwrites_existing_artifact(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    store.put_model_version(
        model_version_id,
        BytesIO(b"first-content"),
    )
    store.put_model_version(
        model_version_id,
        BytesIO(b"second-content"),
    )

    file_path, _ = store._build_model_version_file_path(model_version_id)

    assert file_path.read_bytes() == b"second-content"


def test_get_model_version_path_returns_artifact_path(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    store.put_model_version(
        model_version_id,
        BytesIO(b"content"),
    )

    expected_path, _ = store._build_model_version_file_path(model_version_id)

    with store.get_model_version_path(model_version_id) as path:
        assert path == expected_path
        assert path.read_bytes() == b"content"


def test_existence_check_creates_lock_file(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    _, lock_path = store._build_model_version_file_path(model_version_id)

    assert not lock_path.exists()

    assert not store.check_model_version_existance(model_version_id)

    assert lock_path.is_file()
