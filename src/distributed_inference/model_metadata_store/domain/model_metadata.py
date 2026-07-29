from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: UserId
    model_id: ModelId
    name: str

    @override
    def model_post_init(
        self,
        __context: Any,
    ) -> None:

        if self.model_id.user_id != self.owner_id:
            raise ValueError(
                f"Model {self.model_id} is not owned by user {self.owner_id}"
            )
        pass


class ModelVersionMetadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    model_id: ModelId = Field(frozen=True)
    model_version_id: ModelVersionId = Field(frozen=True)

    version_number: int = Field(frozen=True)

    model_info: ModelInfo = Field(frozen=True)
    model_graph: ModelGraph | None = None

    @override
    def model_post_init(
        self,
        __context: Any,
    ) -> None:

        if self.model_version_id.model_id != self.model_id:
            raise ValueError(
                f"Model version {self.model_version_id} does not belong to model {self.model_id}"
            )

        if self.model_version_id.version_number != self.version_number:
            raise ValueError(
                f"Model version {self.model_version_id} has version number {self.version_number} but should have version number {self.model_version_id.version_number}"
            )

        if self.model_graph is not None:
            self._validate_graph(self.model_graph)

        pass

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if self.model_graph is not None:
            self._validate_graph(self.model_graph)

        return self

    def _validate_graph(
        self,
        model_graph: ModelGraph,
    ) -> None:

        if model_graph.get_model_info() != self.model_info:
            raise ValueError(
                f"Model version {self.model_version_id} has model info {self.model_info} but model graph has model info {self.model_graph.get_model_info()}"
            )

        if model_graph.get_model_info() is None:
            raise ValueError(
                f"Model graph for model version {self.model_version_id} has no model info"
            )

        if model_graph.get_model_info() != self.model_info:
            raise ValueError(
                f"Model version {self.model_version_id} has model info {self.model_info} but model graph has model info {model_graph.get_model_info()}"
            )


class SubModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
