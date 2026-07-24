from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)


class SubModelArtifactStoreContract(ABC):
    @abstractmethod
    def build_store(
        self,
        base_path: Path,
    ) -> SubModelArtifactStore:
        raise NotImplementedError

    @pytest.fixture
    def store(
        self,
        tmp_path: Path,
    ) -> SubModelArtifactStore:
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

    @pytest.fixture
    def sub_model_id(
        self,
        model_version_id: ModelVersionId,
    ) -> SubModelId:
        return SubModelId(
            model_version_id=model_version_id,
            layers=(
                "layer_1",
                "layer_2",
            ),
        )

    def test_missing_submodel_does_not_exist(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        assert not store.check_sub_model_existance(sub_model_id)

    def test_put_submodel_makes_artifact_exist(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            BytesIO(b"onnx-submodel-content"),
        )

        assert store.check_sub_model_existance(sub_model_id)

    def test_get_submodel_returns_stored_content(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        expected = b"onnx-submodel-content"

        store.put_sub_model(
            sub_model_id,
            BytesIO(expected),
        )

        with store.get_sub_model(sub_model_id) as stream:
            assert stream.read() == expected

    def test_get_submodel_closes_stream_after_context(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            BytesIO(b"content"),
        )

        with store.get_sub_model(sub_model_id) as stream:
            assert not stream.closed

        assert stream.closed

    def test_get_missing_submodel_raises(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            with store.get_sub_model(sub_model_id):
                pass

    def test_different_submodels_are_independent(
        self,
        store: SubModelArtifactStore,
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

        store.put_sub_model(
            first_id,
            BytesIO(b"first-submodel"),
        )
        store.put_sub_model(
            second_id,
            BytesIO(b"second-submodel"),
        )

        with store.get_sub_model(first_id) as first:
            assert first.read() == b"first-submodel"

        with store.get_sub_model(second_id) as second:
            assert second.read() == b"second-submodel"
