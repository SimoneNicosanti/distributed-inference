from abc import ABC, abstractmethod
from collections.abc import Iterable

from artifacts.contracts.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.domain.model_variant_graph import ModelVariantGraph
from shared.model.keys import LayerKey


class ModelPartitionExtractor(ABC):
    @abstractmethod
    async def extract_model_partition(
        self,
        model_graph: ModelVariantGraph,
        layers: Iterable[LayerKey],
        input_paths: ArtifactWorkspace,
        output_paths: ArtifactWorkspace,
    ) -> None: ...
