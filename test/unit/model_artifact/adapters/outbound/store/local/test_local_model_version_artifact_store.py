from pathlib import Path
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from distributed_inference.model_artifact.adapters.outbound.local import (
    local_storage_bundle_utils,
)
from distributed_inference.model_artifact.adapters.outbound.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from test.support.model_artifact.artifact_bundle_test_utils import build_test_bundle


@pytest.fixture
def store(
    tmp_path: Path,
) -> LocalModelVersionArtifactStore:
    return LocalModelVersionArtifactStore(tmp_path)


@pytest.fixture
def model_version_id() -> ModelVersionId:
    model_id = ModelId(
        user_id=UserId(user_id=uuid4()),
        model_name="resnet50",
    )

    return ModelVersionId(
        model_id=model_id,
        version_number=3,
    )


def test_constructor_creates_storage_directories(
    tmp_path: Path,
) -> None:
    store = LocalModelVersionArtifactStore(tmp_path)

    assert store.base_path == tmp_path
    assert store.model_versions_dir == (tmp_path / "model_versions")
    assert store.lock_dir == (tmp_path / "model_versions" / ".lock")

    assert store.base_path.is_dir()
    assert store.model_versions_dir.is_dir()
    assert store.lock_dir.is_dir()


def test_bundle_root_is_specific_to_model_version(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    root_path, _ = store._build_bundle_root_path_and_lock_file(model_version_id)

    expected_parent = (
        store.model_versions_dir
        / str(model_version_id.model_id.user_id.user_id)
        / model_version_id.model_id.model_name
    )

    assert root_path.parent == expected_parent
    assert str(model_version_id.version_number) in root_path.name


def test_different_versions_have_different_bundle_roots(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    second_version_id = ModelVersionId(
        model_id=model_version_id.model_id,
        version_number=4,
    )

    first_root, _ = store._build_bundle_root_path_and_lock_file(model_version_id)
    second_root, _ = store._build_bundle_root_path_and_lock_file(second_version_id)

    assert first_root != second_root


def test_lock_path_is_specific_to_model_version(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    _, lock_path = store._build_bundle_root_path_and_lock_file(model_version_id)

    assert lock_path.parent == store.lock_dir
    assert lock_path.suffix == ".lock"
    assert str(model_version_id.model_id.user_id) in lock_path.name
    assert model_version_id.model_id.model_name in lock_path.name
    assert str(model_version_id.version_number) in lock_path.name


def test_put_bundle_creates_all_bundle_files(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(model_version_id)

    store.put_model_version(
        model_version_id,
        build_test_bundle(
            model_content=b"model-content",
            weights_content=b"weights-content",
        ),
    )

    assert (bundle_root_path / "model.onnx").read_bytes() == b"model-content"

    assert (
        bundle_root_path / "weights" / "model.data"
    ).read_bytes() == b"weights-content"


def test_put_bundle_creates_manifest(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(model_version_id)

    store.put_model_version(
        model_version_id,
        build_test_bundle(),
    )

    manifest_path = bundle_root_path / local_storage_bundle_utils.MANIFEST_FILE_NAME

    assert manifest_path.is_file()
    assert manifest_path.read_text(encoding="utf-8")


def test_put_bundle_creates_lock_file(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    _, lock_path = store._build_bundle_root_path_and_lock_file(model_version_id)

    assert not lock_path.exists()

    store.put_model_version(
        model_version_id,
        build_test_bundle(),
    )

    assert lock_path.is_file()


def test_check_existence_creates_lock_file(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    _, lock_path = store._build_bundle_root_path_and_lock_file(model_version_id)

    assert not lock_path.exists()

    assert not store.check_model_version_existence(model_version_id)

    assert lock_path.is_file()


def test_materialized_bundle_contains_local_root_and_entrypoint(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    store.put_model_version(
        model_version_id,
        build_test_bundle(
            model_content=b"model-content",
        ),
    )

    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(model_version_id)
    expected_entrypoint = bundle_root_path / "model.onnx"

    with store.get_model_version_bundle_path(model_version_id) as concrete_paths:
        assert concrete_paths.root_path == bundle_root_path
        assert concrete_paths.entrypoint_path == expected_entrypoint
        assert expected_entrypoint.read_bytes() == b"model-content"


def test_materialized_bundle_points_to_existing_nested_files(
    store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    store.put_model_version(
        model_version_id,
        build_test_bundle(
            weights_content=b"external-data",
        ),
    )

    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(model_version_id)

    with store.get_model_version_bundle_path(model_version_id):
        assert (
            bundle_root_path / "weights" / "model.data"
        ).read_bytes() == b"external-data"
