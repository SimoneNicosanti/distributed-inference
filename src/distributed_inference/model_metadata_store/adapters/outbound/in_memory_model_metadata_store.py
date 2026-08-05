from typing import override

from distributed_inference.model_manager.domain.model import Model, ModelId
from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ModelVersionId,
    ProfiledModelVersion,
)
from distributed_inference.model_manager.domain.sub_model import SubModel, SubModelId
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)


class InMemoryModelMetadataStore(ModelMetadataStore):
    ## In this case, we need no lock: since there is no await in the methods
    ## No coroutine can interleave the execution, so the access will always be safe
    def __init__(self) -> None:
        self._models: dict[ModelId, Model] = {}
        self._model_versions: dict[ModelVersionId, ModelVersion] = {}
        self._profiled_model_versions: dict[ModelVersionId, ProfiledModelVersion] = {}
        self._sub_models: dict[SubModelId, SubModel] = {}

    @override
    async def register_model(self, model: Model) -> ModelId:

        model_id = model.model_id
        if model_id in self._models.keys():
            ## We do not allow registering the same model twice
            raise ValueError(f"Model {model_id} already exists")
        self._models[model_id] = model

        return model_id

    @override
    async def get_model(self, model_id: ModelId) -> Model:
        if model_id not in self._models.keys():
            raise ValueError(f"Model {model_id} does not exist")
        return self._models[model_id]

    @override
    async def check_model_existence(self, model_id: ModelId) -> bool:
        return model_id in self._models.keys()

    @override
    async def register_model_version(
        self,
        model_version: ModelVersion,
    ) -> ModelVersionId:

        model_id = model_version.model_id
        if model_id not in self._models.keys():
            raise ValueError(f"Model {model_id} does not exist")

        model_version_id = model_version.model_version_id
        if model_version_id in self._model_versions.keys():
            ## We do not allow registering the same model version twice
            raise ValueError(f"Model version {model_version_id} already exists")
        self._model_versions[model_version_id] = model_version

        return model_version_id

    @override
    async def get_model_version(self, model_version_id: ModelVersionId) -> ModelVersion:
        if model_version_id not in self._model_versions.keys():
            raise ValueError(f"Model version {model_version_id} does not exist")
        return self._model_versions[model_version_id]

    @override
    async def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool:
        return model_version_id in self._model_versions.keys()

    @override
    async def register_profiled_model_version(
        self,
        profiled_model_version: ProfiledModelVersion,
    ) -> ModelVersionId:

        model_version_id = profiled_model_version.model_version_id
        if model_version_id not in self._model_versions.keys():
            raise ValueError(f"Model version {model_version_id} does not exist")

        if model_version_id in self._profiled_model_versions.keys():
            ## Idempotence
            return model_version_id

        self._profiled_model_versions[model_version_id] = profiled_model_version

        return model_version_id

    @override
    async def get_profiled_model_version(
        self, model_version_id: ModelVersionId
    ) -> ProfiledModelVersion | None:
        if model_version_id not in self._profiled_model_versions.keys():
            return None
        return self._profiled_model_versions[model_version_id]

    @override
    async def check_profiled_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool:
        return model_version_id in self._profiled_model_versions.keys()

    @override
    async def register_sub_model(
        self,
        sub_model: SubModel,
    ) -> SubModelId:

        sub_model_id = sub_model.sub_model_id
        model_version_id = sub_model_id.model_version_id
        if model_version_id not in self._profiled_model_versions.keys():
            raise ValueError(f"Model version {model_version_id} does not exist")

        if sub_model_id in self._sub_models.keys():
            ## Idempotence
            return sub_model_id

        self._sub_models[sub_model_id] = sub_model

        return sub_model_id

    @override
    async def get_sub_model(self, sub_model_id: SubModelId) -> SubModel:
        if sub_model_id not in self._sub_models.keys():
            raise ValueError(f"Sub model {sub_model_id} does not exist")
        return self._sub_models[sub_model_id]

    @override
    async def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool:
        return sub_model_id in self._sub_models.keys()
