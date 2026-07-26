from contextlib import AbstractContextManager
from typing import cast
from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

from distributed_inference.adapters.outbound.model_artifact.materializer.local.local_sub_model_materializer import (
    LocalSubModelMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)


def build_sub_model_id() -> SubModelId:
    model_id = ModelId(
        user_id=UserId(user_id=uuid4()),
        model_name="resnet50",
    )

    return SubModelId(
        model_version_id=ModelVersionId(
            model_id=model_id,
            version_number=1,
        ),
        layers=(
            "layer_1",
            "layer_2",
        ),
    )


def test_constructor_stores_sub_model_store() -> None:
    raw_store_mock = create_autospec(
        LocalSubModelArtifactStore,
        instance=True,
    )
    store = cast(
        LocalSubModelArtifactStore,
        raw_store_mock,
    )

    materializer = LocalSubModelMaterializer(store)

    assert materializer._local_sub_model_artifact_store is store


def test_materialize_sub_model_delegates_to_store() -> None:
    raw_store_mock = create_autospec(
        LocalSubModelArtifactStore,
        instance=True,
    )
    store = cast(
        LocalSubModelArtifactStore,
        raw_store_mock,
    )

    expected_context_manager = cast(
        AbstractContextManager[ArtifactConcretePaths],
        MagicMock(),
    )

    raw_store_mock.get_sub_model_path.return_value = expected_context_manager

    materializer = LocalSubModelMaterializer(store)
    sub_model_id = build_sub_model_id()

    result = materializer.materialize_sub_model(sub_model_id)

    assert result is expected_context_manager

    raw_store_mock.get_sub_model_path.assert_called_once_with(sub_model_id)
