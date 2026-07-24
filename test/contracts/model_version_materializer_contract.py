from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

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


class ModelVersionMaterializerContract(ABC):
    @abstractmethod
    def build_backend(
        self,
        base_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        raise NotImplementedError

    @pytest.fixture
    def backend(
        self,
        tmp_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        return self.build_backend(tmp_path)

    @pytest.fixture
    def model_version_id(self) -> ModelVersionId:
        owner_id = UserId(user_id=uuid4())

        model_id = ModelId(
            user_id=owner_id,
            model_name="resnet50",
        )

        return ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

    def test_materialized_model_version_exists(
        self,
        backend: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, artifact_store = backend

        artifact_store.put_model_version(
            model_version_id,
            BytesIO(b"model-content"),
        )

        with materializer.materialize_model_version(model_version_id) as model_path:
            assert model_path.exists()
            assert model_path.is_file()

    def test_materialized_model_version_contains_stored_content(
        self,
        backend: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, artifact_store = backend
        expected = b"onnx-model-content"

        artifact_store.put_model_version(
            model_version_id,
            BytesIO(expected),
        )

        with materializer.materialize_model_version(model_version_id) as model_path:
            assert model_path.read_bytes() == expected

    def test_model_version_can_be_materialized_multiple_times(
        self,
        backend: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, artifact_store = backend
        expected = b"model-content"

        artifact_store.put_model_version(
            model_version_id,
            BytesIO(expected),
        )

        with materializer.materialize_model_version(model_version_id) as first_path:
            assert first_path.read_bytes() == expected

        with materializer.materialize_model_version(model_version_id) as second_path:
            assert second_path.read_bytes() == expected

    def test_different_versions_materialize_different_contents(
        self,
        backend: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, artifact_store = backend

        second_version_id = ModelVersionId(
            model_id=model_version_id.model_id,
            version_number=2,
        )

        artifact_store.put_model_version(
            model_version_id,
            BytesIO(b"version-one"),
        )
        artifact_store.put_model_version(
            second_version_id,
            BytesIO(b"version-two"),
        )

        with materializer.materialize_model_version(model_version_id) as first_path:
            assert first_path.read_bytes() == b"version-one"

        with materializer.materialize_model_version(second_version_id) as second_path:
            assert second_path.read_bytes() == b"version-two"
