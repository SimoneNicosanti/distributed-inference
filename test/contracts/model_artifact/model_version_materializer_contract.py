from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from distributed_inference.model_materializer.application.ports.outbound.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.model_artifact.application.ports.outbound.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from test.support.model_artifact.artifact_bundle_test_utils import (
    build_test_bundle,
)
from test.support.model_artifact.materializer_test_utils import (
    extract_root_and_entrypoint,
)


class ModelVersionMaterializerContract(ABC):
    @abstractmethod
    def build_dependencies(
        self,
        base_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        raise NotImplementedError

    @pytest.fixture
    def dependencies(
        self,
        tmp_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        return self.build_dependencies(tmp_path)

    @pytest.fixture
    def model_version_id(self) -> ModelVersionId:
        return ModelVersionId(
            model_id=ModelId(
                user_id=UserId(user_id=uuid4()),
                model_name="resnet50",
            ),
            version_number=1,
        )

    def test_materialize_missing_model_version_raises(
        self,
        dependencies: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, _ = dependencies

        with pytest.raises(FileNotFoundError):
            with materializer.materialize_model_version(model_version_id):
                pass

    def test_materialize_returns_root_and_entrypoint(
        self,
        dependencies: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, store = dependencies

        bundle = build_test_bundle(
            model_content=b"model-content",
            weights_content=b"weights-content",
        )

        store.put_model_version(
            model_version_id,
            bundle,
        )

        with materializer.materialize_model_version(model_version_id) as concrete_paths:
            root_path, entrypoint_path = extract_root_and_entrypoint(concrete_paths)

            expected_entrypoint = root_path.joinpath(
                *bundle.manifest.rel_entrypoint_path.parts
            )

            assert root_path.is_dir()
            assert entrypoint_path.is_file()
            assert entrypoint_path == expected_entrypoint
            assert entrypoint_path.is_relative_to(root_path)

    def test_materialized_root_contains_all_artifact_files(
        self,
        dependencies: tuple[
            ModelVersionMaterializer,
            ModelVersionArtifactStore,
        ],
        model_version_id: ModelVersionId,
    ) -> None:
        materializer, store = dependencies

        bundle = build_test_bundle(
            model_content=b"model-content",
            weights_content=b"weights-content",
        )

        store.put_model_version(
            model_version_id,
            bundle,
        )

        with materializer.materialize_model_version(model_version_id) as concrete_paths:
            root_path, _ = extract_root_and_entrypoint(concrete_paths)

            model_path = root_path / "model.onnx"
            weights_path = root_path / "weights" / "model.data"

            assert model_path.read_bytes() == b"model-content"
            assert weights_path.read_bytes() == b"weights-content"
