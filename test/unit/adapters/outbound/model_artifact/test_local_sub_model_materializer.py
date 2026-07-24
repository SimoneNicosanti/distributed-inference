from io import BytesIO
from pathlib import Path
from typing import override
from uuid import uuid4

import pytest

from distributed_inference.adapters.outbound.model_artifact.materializer.local_sub_model_materializer import (
    LocalSubModelMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.materializer.sub_model_materializer import (
    SubModelMaterializer,
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
from test.contracts.sub_model_materializer_contract import (
    SubModelMaterializerContract,
)


class TestLocalSubModelMaterializerContract(SubModelMaterializerContract):
    @override
    def build_backend(
        self,
        base_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
    ]:
        artifact_store = LocalSubModelArtifactStore(base_path)

        materializer = LocalSubModelMaterializer(artifact_store)

        return materializer, artifact_store


@pytest.fixture
def artifact_store(
    tmp_path: Path,
) -> LocalSubModelArtifactStore:
    return LocalSubModelArtifactStore(tmp_path)


@pytest.fixture
def materializer(
    artifact_store: LocalSubModelArtifactStore,
) -> LocalSubModelMaterializer:
    return LocalSubModelMaterializer(artifact_store)


@pytest.fixture
def sub_model_id() -> SubModelId:
    owner_id = UserId(user_id=uuid4())

    model_id = ModelId(
        user_id=owner_id,
        model_name="resnet50",
    )

    model_version_id = ModelVersionId(
        model_id=model_id,
        version_number=3,
    )

    return SubModelId(
        model_version_id=model_version_id,
        layers=(
            "layer_1",
            "layer_2",
        ),
    )


def test_materializer_keeps_artifact_store_reference(
    materializer: LocalSubModelMaterializer,
    artifact_store: LocalSubModelArtifactStore,
) -> None:
    assert materializer._local_sub_model_artifact_store is artifact_store


def test_materializer_returns_local_artifact_path(
    materializer: LocalSubModelMaterializer,
    artifact_store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    artifact_store.put_sub_model(
        sub_model_id,
        BytesIO(b"content"),
    )

    expected_path, _ = artifact_store._build_sub_model_file_path(sub_model_id)

    with materializer.materialize_sub_model(sub_model_id) as materialized_path:
        assert materialized_path == expected_path


def test_local_materialization_does_not_copy_artifact(
    materializer: LocalSubModelMaterializer,
    artifact_store: LocalSubModelArtifactStore,
    sub_model_id: SubModelId,
) -> None:
    artifact_store.put_sub_model(
        sub_model_id,
        BytesIO(b"content"),
    )

    expected_path, _ = artifact_store._build_sub_model_file_path(sub_model_id)

    with materializer.materialize_sub_model(sub_model_id) as materialized_path:
        assert materialized_path.samefile(expected_path)
