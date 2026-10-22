from fastapi import APIRouter

from integration.model_manager.http.model_manager_http_paths import (
    FINALIZE_MODEL_VARIANT_REGISTRATION_PATH,
    GENERATE_MODEL_PARTITION_PATH,
    REGISTER_MODEL_PATH,
    START_MODEL_VARIANT_REGISTRATION_PATH,
)
from integration.model_manager.http.model_manager_http_schema import (
    FinalizeModelVariantRegistrationRequest,
    GenerateModelPartitionRequest,
    GenerateModelPartitionResponse,
    RegisterModelRequest,
    StartModelVariantRegistrationRequest,
    StartModelVariantRegistrationResponse,
)
from model_manager.application.ports.inbound.model_partitioner import ModelPartitioner
from model_manager.application.ports.inbound.model_registry import ModelRegistry
from model_manager.application.ports.inbound.model_variant_registry import (
    ModelVariantRegistry,
)
from model_manager.domain.model import Model, ModelInfo
from model_manager.domain.model_variant import ModelVariant, ModelVariantInfo

CHUNK_SIZE = 1024 * 1024


def build_model_manager_router(
    model_registry: ModelRegistry,
    model_variant_registry: ModelVariantRegistry,
    model_partitioner: ModelPartitioner,
) -> APIRouter:

    router = APIRouter(
        tags=["model-manager"],
    )

    @router.post(
        path=REGISTER_MODEL_PATH,
    )
    async def register_model(
        request: RegisterModelRequest,
    ) -> None:
        model_dto = request.model_dto

        model_info = ModelInfo(
            model_task=model_dto.model_task,
            architecture_info=model_dto.architecture_info,
        )
        model = Model(
            model_id=model_dto.model_id,
            visibility=model_dto.visibility,
            model_info=model_info,
        )

        await model_registry.register_model(model=model)

    @router.post(
        path=START_MODEL_VARIANT_REGISTRATION_PATH,
        response_model=StartModelVariantRegistrationResponse,
    )
    async def start_model_variant_registration(
        request: StartModelVariantRegistrationRequest,
    ) -> StartModelVariantRegistrationResponse:

        model_variant_dto = request.model_variant_dto

        model_variant_info = ModelVariantInfo.model_validate(
            model_variant_dto.model_dump(exclude={"id"})
        )

        model_variant = ModelVariant(
            model_variant_id=model_variant_dto.id,
            model_variant_info=model_variant_info,
        )

        artifact_ref = await model_variant_registry.start_model_variant_registration(
            model_variant=model_variant
        )

        return StartModelVariantRegistrationResponse(
            artifact_ref=artifact_ref,
        )

    @router.post(
        path=FINALIZE_MODEL_VARIANT_REGISTRATION_PATH,
    )
    async def finalize_model_variant_registration(
        request: FinalizeModelVariantRegistrationRequest,
    ) -> None:

        model_variant_id = request.model_variant_id

        await model_variant_registry.finalize_model_variant_registration(
            model_variant_id=model_variant_id
        )

    @router.post(
        path=GENERATE_MODEL_PARTITION_PATH,
        response_model=GenerateModelPartitionResponse,
    )
    async def generate_model_variant_partition(
        request: GenerateModelPartitionRequest,
    ) -> GenerateModelPartitionResponse:

        model_variant_id = request.model_variant_id
        layers = request.layers

        model_partition = await model_partitioner.generate_model_variant_partition(
            model_variant_id=model_variant_id,
            layers=layers,
        )

        return GenerateModelPartitionResponse(
            model_partition_id=model_partition.model_partition_id,
            artifact_ref=model_partition.artifact_ref,
        )

    return router
