import asyncio
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, UploadFile

from distributed_inference.artifact_processing import (
    compression_utils,
)
from distributed_inference.model_manager.adapters.inbound.http.model_manager_http_schema import (
    GenerateSubModelRequest,
    GenerateSubModelResponse,
    GetProfiledModelVersionRequest,
    GetProfiledModelVersionResponse,
    RegisterModelRequest,
    RegisterModelResponse,
    UploadModelVersionResponse,
)
from distributed_inference.model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from distributed_inference.model_manager.domain.model_version import ModelVersion

CHUNK_SIZE = 1024 * 1024


def build_model_manager_router(
    model_manager: ModelManager,
) -> APIRouter:

    router = APIRouter(
        prefix="/model-manager",
        tags=["model-manager"],
    )

    @router.post(
        "/models",
        response_model=RegisterModelResponse,
    )
    async def register_model(
        request: RegisterModelRequest,
    ) -> RegisterModelResponse:
        model_id = await model_manager.register_model(model=request.model)

        return RegisterModelResponse(model_id=model_id)

    @router.post(
        "/model-versions",
        response_model=UploadModelVersionResponse,
    )
    async def upload_model_version(
        model_version_json: str = Form(),
        bundle_zip: UploadFile = File(),
    ) -> UploadModelVersionResponse:
        model_version = ModelVersion.model_validate_json(model_version_json)

        async with aiofiles.tempfile.NamedTemporaryFile() as zip_file:
            zip_file_path = Path(str(zip_file.name))
            while chunk := await asyncio.to_thread(bundle_zip.file.read, CHUNK_SIZE):
                await zip_file.write(chunk)

            async with compression_utils.decompress_artifact_bundle(
                zip_file_path,
            ) as artifact_bundle:
                model_version_id = await model_manager.upload_model_version(
                    model_version=model_version,
                    bundle=artifact_bundle,
                )

        return UploadModelVersionResponse(
            model_version_id=model_version_id,
        )

    @router.post(
        "/sub-models",
        response_model=GenerateSubModelResponse,
    )
    async def generate_sub_model(
        request: GenerateSubModelRequest,
    ) -> GenerateSubModelResponse:
        sub_model = await model_manager.generate_sub_model(
            model_version_id=request.model_version_id,
            layers=request.layers,
        )

        return GenerateSubModelResponse(
            sub_model=sub_model,
        )

    @router.get(
        "/model-versions/profiled", response_model=GetProfiledModelVersionResponse
    )
    async def get_profiled_model_version(
        request: GetProfiledModelVersionRequest,
    ) -> GetProfiledModelVersionResponse:
        profiled_model_version = await model_manager.get_profiled_model_version(
            request.model_version_id
        )

        return GetProfiledModelVersionResponse(
            profiled_model_version=profiled_model_version
        )

    return router
