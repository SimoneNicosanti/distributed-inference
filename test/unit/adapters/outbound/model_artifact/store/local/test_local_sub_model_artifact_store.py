from pathlib import Path

import pytest

from distributed_inference.adapters.outbound.model_artifact.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MANIFEST_FILE_NAME,
)
from distributed_inference.domain.identifiers import SubModelId
from test.contracts.sub_model_artifact_store_contract import (
    build_layers,
    build_sub_model_id,
)
from test.support.artifact_bundle_test_utils import build_test_bundle


@pytest.fixture
def store(
    tmp_path: Path,
) -> LocalSubModelArtifactStore:
    return LocalSubModelArtifactStore(tmp_path)


@pytest.fixture
def sub_model_id() -> SubModelId:
    return build_sub_model_id(
        layers=build_layers("layer_1", "layer_2"),
    )


def test_constructor_creates_storage_directories(
    tmp_path: Path,
) -> None:
    store = LocalSubModelArtifactStore(tmp_path)

    assert store.base_path == tmp_path
    assert store.sub_models_dir == tmp_path / "sub_models"
    assert store.lock_dir == tmp_path / "sub_models" / ".lock"

    assert store.base_path.is_dir()
    assert store.sub_models_dir.is_dir()
    assert store.lock_dir.is_dir()


def test_bundle_root_is_specific_to_layer_set(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    root_path, _ = store._build_bundle_root_path_and_lock_file(sub_model_id)

    model_id = sub_model_id.model_version_id.model_id
    expected_parent = (
        store.sub_models_dir
        / str(model_id.user_id.user_id)
        / model_id.model_name
        / str(sub_model_id.model_version_id.version_number)
        / store._hash_layers(sub_model_id.layers)
    )

    assert root_path == expected_parent
    assert root_path.name == store._hash_layers(sub_model_id.layers)


def test_different_layer_sets_have_different_bundle_roots(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    second_sub_model_id = SubModelId(
        model_version_id=sub_model_id.model_version_id,
        layers=build_layers("layer_3", "layer_4"),
    )

    first_root, _ = store._build_bundle_root_path_and_lock_file(sub_model_id)
    second_root, _ = store._build_bundle_root_path_and_lock_file(second_sub_model_id)

    assert first_root != second_root


def test_lock_path_is_specific_to_layer_set(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    _, lock_path = store._build_bundle_root_path_and_lock_file(sub_model_id)

    model_id = sub_model_id.model_version_id.model_id
    layers_hash = store._hash_layers(sub_model_id.layers)

    assert lock_path.parent == store.lock_dir
    assert lock_path.suffix == ".lock"
    assert str(model_id.user_id.user_id) in lock_path.name
    assert model_id.model_name in lock_path.name
    assert str(sub_model_id.model_version_id.version_number) in (lock_path.name)
    assert layers_hash in lock_path.name


def test_layers_hash_is_deterministic(
    store: LocalSubModelArtifactStore,
) -> None:
    layers = build_layers("layer_1", "layer_2")

    assert store._hash_layers(layers) == store._hash_layers(layers)


def test_layer_order_does_not_change_bundle_root(
    store: LocalSubModelArtifactStore,
) -> None:
    first = build_sub_model_id(
        layers=build_layers("layer_1", "layer_2"),
    )
    second = SubModelId(
        model_version_id=first.model_version_id,
        layers=build_layers("layer_2", "layer_1"),
    )

    first_root, _ = store._build_bundle_root_path_and_lock_file(first)
    second_root, _ = store._build_bundle_root_path_and_lock_file(second)

    assert first_root == second_root


def test_put_bundle_creates_all_bundle_files(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(sub_model_id)

    store.put_sub_model(
        sub_model_id,
        build_test_bundle(
            model_content=b"sub-model-content",
            weights_content=b"sub-model-weights",
        ),
    )

    assert (bundle_root_path / "model.onnx").read_bytes() == b"sub-model-content"

    assert (
        bundle_root_path / "weights" / "model.data"
    ).read_bytes() == b"sub-model-weights"


def test_put_bundle_creates_manifest(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(sub_model_id)

    store.put_sub_model(
        sub_model_id,
        build_test_bundle(),
    )

    manifest_path = bundle_root_path / MANIFEST_FILE_NAME

    assert manifest_path.is_file()
    assert manifest_path.read_text(encoding="utf-8")


def test_check_existence_creates_lock_file(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    _, lock_path = store._build_bundle_root_path_and_lock_file(sub_model_id)

    assert not lock_path.exists()

    assert not store.check_sub_model_existence(sub_model_id)

    assert lock_path.is_file()


def test_materialized_bundle_contains_root_and_entrypoint(
    store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    store.put_sub_model(
        sub_model_id,
        build_test_bundle(model_content=b"sub-model"),
    )

    bundle_root_path, _ = store._build_bundle_root_path_and_lock_file(sub_model_id)
    expected_entrypoint = bundle_root_path / "model.onnx"

    with store.get_sub_model_path(sub_model_id) as concrete_paths:
        assert concrete_paths.root_path == bundle_root_path
        assert concrete_paths.entrypoint_path == expected_entrypoint
        assert expected_entrypoint.read_bytes() == b"sub-model"
