from distributed_inference.domain.identifiers import ModelVersionId
from distributed_inference.domain.model_graph_info import ModelGraph

from abc import ABC, abstractmethod


class ModelProfile(ABC):
    @abstractmethod
    def profile_model(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph: ...

    @abstractmethod
    def profile_model_with_no_optimization(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph: ...

    @abstractmethod
    def profile_model_with_optimization(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph: ...

    @abstractmethod
    def profile_model_with_aggregation(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph: ...
