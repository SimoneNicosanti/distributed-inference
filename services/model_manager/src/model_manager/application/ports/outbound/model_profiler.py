from abc import ABC, abstractmethod

from artifacts.workspace.artifact_workspace import ArtifactWorkspace
from integration.model.model import ModelInfo
from integration.model.model_version import (
    ModelVersion,
)
from model_manager.domain.profiled_model_version import ProfiledModelVersion


class ModelProfiler(ABC):
    @abstractmethod
    async def profile_model_version(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_version: ModelVersion,
    ) -> ProfiledModelVersion: ...
