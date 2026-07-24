from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

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


class SubModelMaterializerContract(ABC):
    @abstractmethod
    def build_backend(
        self,
        base_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
    ]:
        raise NotImplementedError

    @pytest.fixture
    def backend(
        self,
        tmp_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
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

    def test_materialized_submodel_exists(
        self,
        backend: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, artifact_store = backend

        artifact_store.put_sub_model(
            sub_model_id,
            BytesIO(b"submodel-content"),
        )

        with materializer.materialize_sub_model(sub_model_id) as sub_model_path:
            assert sub_model_path.exists()
            assert sub_model_path.is_file()

    def test_materialized_submodel_contains_stored_content(
        self,
        backend: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, artifact_store = backend
        expected = b"onnx-submodel-content"

        artifact_store.put_sub_model(
            sub_model_id,
            BytesIO(expected),
        )

        with materializer.materialize_sub_model(sub_model_id) as sub_model_path:
            assert sub_model_path.read_bytes() == expected

    def test_submodel_can_be_materialized_multiple_times(
        self,
        backend: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, artifact_store = backend
        expected = b"submodel-content"

        artifact_store.put_sub_model(
            sub_model_id,
            BytesIO(expected),
        )

        with materializer.materialize_sub_model(sub_model_id) as first_path:
            assert first_path.read_bytes() == expected

        with materializer.materialize_sub_model(sub_model_id) as second_path:
            assert second_path.read_bytes() == expected

    def test_different_submodels_materialize_different_contents(
        self,
        backend: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, artifact_store = backend

        first_id = SubModelId(
            model_version_id=model_version_id,
            layers=("layer_1",),
        )
        second_id = SubModelId(
            model_version_id=model_version_id,
            layers=("layer_2",),
        )

        artifact_store.put_sub_model(
            first_id,
            BytesIO(b"first-submodel"),
        )
        artifact_store.put_sub_model(
            second_id,
            BytesIO(b"second-submodel"),
        )

        with materializer.materialize_sub_model(first_id) as first_path:
            assert first_path.read_bytes() == b"first-submodel"

        with materializer.materialize_sub_model(second_id) as second_path:
            assert second_path.read_bytes() == b"second-submodel"
