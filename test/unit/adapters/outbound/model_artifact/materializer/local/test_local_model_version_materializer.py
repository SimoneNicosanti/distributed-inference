from contextlib import AbstractContextManager
from typing import cast
from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

from distributed_inference.adapters.outbound.model_artifact.materializer.local.local_model_version_materializer import (
    LocalModelVersionMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)


def build_model_version_id() -> ModelVersionId:
    return ModelVersionId(
        model_id=ModelId(
            user_id=UserId(user_id=uuid4()),
            model_name="resnet50",
        ),
        version_number=1,
    )


def test_constructor_stores_model_version_store() -> None:
    raw_store_mock = create_autospec(
        LocalModelVersionArtifactStore,
        instance=True,
    )
    store = cast(
        LocalModelVersionArtifactStore,
        raw_store_mock,
    )

    materializer = LocalModelVersionMaterializer(store)

    assert materializer._local_model_version_artifact_store is store


def test_materialize_model_version_delegates_to_store() -> None:
    raw_store_mock = create_autospec(
        LocalModelVersionArtifactStore,
        instance=True,
    )
    store = cast(
        LocalModelVersionArtifactStore,
        raw_store_mock,
    )

    expected_context_manager = cast(
        AbstractContextManager[ArtifactConcretePaths],
        MagicMock(),
    )

    raw_store_mock.get_model_version_bundle_path.return_value = expected_context_manager

    materializer = LocalModelVersionMaterializer(store)
    model_version_id = build_model_version_id()

    result = materializer.materialize_model_version(model_version_id)

    assert result is expected_context_manager

    raw_store_mock.get_model_version_bundle_path.assert_called_once_with(
        model_version_id
    )
