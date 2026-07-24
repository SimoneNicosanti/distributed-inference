from io import BytesIO
from pathlib import Path
from typing import override
from uuid import uuid4

import pytest

from distributed_inference.adapters.outbound.model_artifact.materializer.local_model_version_materializer import (
    LocalModelVersionMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from test.contracts.model_version_materializer_contract import (
    ModelVersionMaterializerContract,
)


class TestLocalModelVersionMaterializerContract(ModelVersionMaterializerContract):
    @override
    def build_backend(
        self,
        base_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        artifact_store = LocalModelVersionArtifactStore(base_path)

        materializer = LocalModelVersionMaterializer(artifact_store)

        return materializer, artifact_store


@pytest.fixture
def artifact_store(
    tmp_path: Path,
) -> LocalModelVersionArtifactStore:
    return LocalModelVersionArtifactStore(tmp_path)


@pytest.fixture
def materializer(
    artifact_store: LocalModelVersionArtifactStore,
) -> LocalModelVersionMaterializer:
    return LocalModelVersionMaterializer(artifact_store)


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


def test_materializer_keeps_artifact_store_reference(
    materializer: LocalModelVersionMaterializer,
    artifact_store: LocalModelVersionArtifactStore,
) -> None:
    assert materializer._local_model_version_artifact_store is artifact_store


def test_materializer_returns_local_artifact_path(
    materializer: LocalModelVersionMaterializer,
    artifact_store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    artifact_store.put_model_version(
        model_version_id,
        BytesIO(b"content"),
    )

    expected_path, _ = artifact_store._build_model_version_file_path(model_version_id)

    with materializer.materialize_model_version(model_version_id) as materialized_path:
        assert materialized_path == expected_path


def test_local_materialization_does_not_copy_artifact(
    materializer: LocalModelVersionMaterializer,
    artifact_store: LocalModelVersionArtifactStore,
    model_version_id: ModelVersionId,
) -> None:
    artifact_store.put_model_version(
        model_version_id,
        BytesIO(b"content"),
    )

    expected_path, _ = artifact_store._build_model_version_file_path(model_version_id)

    with materializer.materialize_model_version(model_version_id) as materialized_path:
        assert materialized_path.samefile(expected_path)
