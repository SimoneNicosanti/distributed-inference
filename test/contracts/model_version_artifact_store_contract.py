from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)


class ModelVersionArtifactStoreContract(ABC):
    @abstractmethod
    def build_store(
        self,
        base_path: Path,
    ) -> ModelVersionArtifactStore:
        raise NotImplementedError

    @pytest.fixture
    def store(
        self,
        tmp_path: Path,
    ) -> ModelVersionArtifactStore:
        return self.build_store(tmp_path)

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

    def test_missing_model_version_does_not_exist(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        assert not store.check_model_version_existance(model_version_id)

    def test_put_model_version_makes_artifact_exist(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            BytesIO(b"onnx-model-content"),
        )

        assert store.check_model_version_existance(model_version_id)

    def test_get_model_version_returns_stored_content(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        expected = b"onnx-model-content"

        store.put_model_version(
            model_version_id,
            BytesIO(expected),
        )

        with store.get_model_version(model_version_id) as stream:
            assert stream.read() == expected

    def test_get_model_version_closes_stream_after_context(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            BytesIO(b"content"),
        )

        with store.get_model_version(model_version_id) as stream:
            assert not stream.closed

        assert stream.closed

    def test_get_missing_model_version_raises(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            with store.get_model_version(model_version_id):
                pass

    def test_different_versions_are_independent(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        second_version_id = ModelVersionId(
            model_id=model_version_id.model_id,
            version_number=2,
        )

        store.put_model_version(
            model_version_id,
            BytesIO(b"version-one"),
        )
        store.put_model_version(
            second_version_id,
            BytesIO(b"version-two"),
        )

        with store.get_model_version(model_version_id) as first:
            assert first.read() == b"version-one"

        with store.get_model_version(second_version_id) as second:
            assert second.read() == b"version-two"
