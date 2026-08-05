from collections.abc import AsyncIterator

import aiofiles
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from distributed_inference.model_manager.adapters.inbound.http import (
    compression_utils,
)
from distributed_inference.model_manager.adapters.inbound.http.schema import (
    DownloadSubModelRequest,
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

        async with compression_utils.decompress_artifact_bundle(
            bundle_zip,
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

    @router.post("/sub-models/download")
    async def download_sub_model(
        request: DownloadSubModelRequest,
    ) -> StreamingResponse:
        async def stream_artifact() -> AsyncIterator[bytes]:
            async with model_manager.download_sub_model(
                request.sub_model_id
            ) as sub_model_bundle:
                async with compression_utils.compress_artifact_bundle(
                    sub_model_bundle
                ) as zip_file_path:
                    async with aiofiles.open(zip_file_path, "rb") as zip_file:
                        while chunk := await zip_file.read(CHUNK_SIZE):
                            yield chunk

        return StreamingResponse(
            stream_artifact(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": ('attachment; filename="artifact.zip"')},
        )

    @router.get("/model-versions/graph", response_model=GetProfiledModelVersionResponse)
    async def get_model_graph(
        request: GetProfiledModelVersionRequest,
    ) -> GetProfiledModelVersionResponse:
        profiled_model_version = await model_manager.get_profiled_model_version(
            request.model_version_id
        )

        return GetProfiledModelVersionResponse(
            profiled_model_version=profiled_model_version
        )

    return router
