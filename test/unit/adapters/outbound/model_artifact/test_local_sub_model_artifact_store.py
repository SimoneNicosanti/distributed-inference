from io import BytesIO
from pathlib import Path
from typing import override
from uuid import uuid4

import pytest

from distributed_inference.adapters.outbound.model_artifact.store.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from test.contracts.sub_model_artifact_store_contract import (
    SubModelArtifactStoreContract,
)


class TestLocalSubModelArtifactStoreContract(SubModelArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> SubModelArtifactStore:
        return LocalSubModelArtifactStore(base_path)


@pytest.fixture
def store(
    tmp_path: Path,
) -> LocalSubModelArtifactStore:
    return LocalSubModelArtifactStore(tmp_path)


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


@pytest.fixture
def sub_model_id(
    model_version_id: ModelVersionId,
) -> SubModelId:
    return SubModelId(
        model_version_id=model_version_id,
        layers=(
            "layer_1",
            "layer_2",
        ),
    )


def test_constructor_creates_required_directories(
    tmp_path: Path,
) -> None:
    store = LocalSubModelArtifactStore(tmp_path)

    assert store.base_path == tmp_path
    assert store.sub_models_dir.is_dir()
    assert store.lock_dir.is_dir()


def test_hash_layers_is_deterministic(
    store: LocalSubModelArtifactStore,
) -> None:
    layers = (
        "layer_1",
        "layer_2",
    )

    first = store._hash_layers(layers)
    second = store._hash_layers(layers)

    assert first == second
    assert len(first) == 32


def test_hash_layers_depends_on_layer_order(
    store: LocalSubModelArtifactStore,
) -> None:
    first = store._hash_layers(
        (
            "layer_1",
            "layer_2",
        )
    )
    second = store._hash_layers(
        (
            "layer_2",
            "layer_1",
        )
    )

    assert first != second


def test_build_submodel_file_path_uses_expected_layout(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    file_path, lock_path = store._build_sub_model_file_path(sub_model_id)

    model_version_id = sub_model_id.model_version_id
    model_id = model_version_id.model_id
    layers_hash = store._hash_layers(sub_model_id.layers)

    assert file_path == (
        store.sub_models_dir
        / str(model_id.user_id)
        / model_id.model_name
        / str(model_version_id.version_number)
        / f"layers_{layers_hash}.onnx"
    )

    assert lock_path == (
        store.lock_dir
        / (
            f"{model_id.user_id}_"
            f"{model_id.model_name}_"
            f"{model_version_id.version_number}_"
            f"{layers_hash}.lock"
        )
    )


def test_put_creates_parent_directories_and_artifact_file(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    file_path, _ = store._build_sub_model_file_path(sub_model_id)

    assert not file_path.parent.exists()

    store.put_sub_model(
        sub_model_id,
        BytesIO(b"content"),
    )

    assert file_path.is_file()
    assert file_path.read_bytes() == b"content"


def test_put_creates_lock_file(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    _, lock_path = store._build_sub_model_file_path(sub_model_id)

    assert not lock_path.exists()

    store.put_sub_model(
        sub_model_id,
        BytesIO(b"content"),
    )

    assert lock_path.is_file()


def test_put_overwrites_existing_artifact(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    store.put_sub_model(
        sub_model_id,
        BytesIO(b"first-content"),
    )
    store.put_sub_model(
        sub_model_id,
        BytesIO(b"second-content"),
    )

    file_path, _ = store._build_sub_model_file_path(sub_model_id)

    assert file_path.read_bytes() == b"second-content"


def test_get_submodel_path_returns_artifact_path(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    store.put_sub_model(
        sub_model_id,
        BytesIO(b"content"),
    )

    expected_path, _ = store._build_sub_model_file_path(sub_model_id)

    with store.get_sub_model_path(sub_model_id) as path:
        assert path == expected_path
        assert path.read_bytes() == b"content"


def test_different_layer_sets_produce_different_paths(
    store: LocalSubModelArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    first_id = SubModelId(
        model_version_id=model_version_id,
        layers=("layer_1",),
    )
    second_id = SubModelId(
        model_version_id=model_version_id,
        layers=("layer_2",),
    )

    first_path, _ = store._build_sub_model_file_path(first_id)
    second_path, _ = store._build_sub_model_file_path(second_id)

    assert first_path != second_path


def test_existence_check_creates_lock_file(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    _, lock_path = store._build_sub_model_file_path(sub_model_id)

    assert not lock_path.exists()

    assert not store.check_sub_model_existance(sub_model_id)

    assert lock_path.is_file()
