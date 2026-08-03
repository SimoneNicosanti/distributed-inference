import hashlib
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from distributed_inference.artifact_store.adapters.outbound.local.local_artifact_store import (
    LocalArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ArtifactKind,
    ModelVersionArtifactKey,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from test.support.artifact_store.artifact_bundle_test_utils import build_test_bundle


def _key() -> ModelVersionArtifactKey:
    return ModelVersionArtifactKey(
        id=ModelVersionId(
            model_id=ModelId(
                user_id=UserId(user_id=uuid4()),
                model_name="resnet50",
            ),
            version_number=3,
        )
    )


@pytest.mark.unit
def test_constructor_creates_directory_for_each_artifact_kind(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    assert store.base_path == tmp_path
    assert all(
        (tmp_path / "artifacts" / kind.value).is_dir() for kind in ArtifactKind
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_put_writes_manifest_and_nested_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    key = _key()
    bundle = build_test_bundle(
        tmp_path / "input",
        files={
            PurePosixPath("model.onnx"): b"model",
            PurePosixPath("weights/model.data"): b"weights",
        },
    )

    await store.put_artifact(key, bundle)

    artifact_root = await store._build_artifact_root_path(key)
    assert (artifact_root / "manifest.json").is_file()
    assert (artifact_root / "model.onnx").read_bytes() == b"model"
    assert (artifact_root / "weights" / "model.data").read_bytes() == b"weights"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifact_root_uses_deterministic_key_digest(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    key = _key()
    expected_digest = hashlib.md5(
        key.model_dump_json().encode("utf-8")
    ).hexdigest()

    artifact_root = await store._build_artifact_root_path(key)

    assert artifact_root == (
        tmp_path / "artifacts" / key.kind.value / expected_digest
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_existence_detects_missing_declared_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    key = _key()
    bundle = build_test_bundle(tmp_path / "input")
    await store.put_artifact(key, bundle)
    artifact_root = await store._build_artifact_root_path(key)
    (artifact_root / "weights" / "model.data").unlink()

    assert not await store.check_artifact_existence(key)


@pytest.mark.unit
def test_local_bundle_rejects_undeclared_file(tmp_path: Path) -> None:
    bundle = build_test_bundle(tmp_path)

    with pytest.raises(FileNotFoundError, match="not declared"):
        bundle.open_file(PurePosixPath("undeclared.bin"))
