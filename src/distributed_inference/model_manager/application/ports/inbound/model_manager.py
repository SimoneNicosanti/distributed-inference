from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Iterable

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    LayerKey,
    ModelGraph,
    ModelInfo,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
)


class ModelManager(ABC):
    @abstractmethod
    def register_model(
        self,
        owner_id: UserId,
        model_name: str,
    ) -> ModelId: ...

    @abstractmethod
    def put_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
        bundle: ArtifactBundle,
    ) -> ModelVersionId: ...

    @abstractmethod
    def generate_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[ArtifactBundle]: ...

    @abstractmethod
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph: ...

    @abstractmethod
    def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...

    @abstractmethod
    def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool: ...
