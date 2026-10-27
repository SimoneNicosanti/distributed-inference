from pathlib import Path
from typing import Annotated

import aiofiles
import aiofiles.tempfile
from fastapi import APIRouter, File, Form, UploadFile

from artifacts.archive.zip_archive import decompress_artifact_bundle
from front_end.application.ports.inbound.inference import (
    InferenceRunner,
    LocalWorkerIdentityQuery,
)
from front_end.application.ports.inbound.model_loader import ModelLoader
from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.dto.worker_identity.worker_identity_dto import WorkerIdentityDto
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOAD_MODEL_VARIANT_PATH,
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
    FRONT_END_REGISTER_MODEL_PATH,
    FRONT_END_RUN_INFERENCE_PATH,
)
from integration.front_end.http.front_end_http_schema import (
    LoadModelVariantResponse,
    RunInferenceRequest,
    RunInferenceResponse,
)
from integration.model_manager.http.model_manager_http_schema import (
    RegisterModelRequest,
    StartModelVariantRegistrationRequest,
)

CHUNK_SIZE = 1024 * 1024


def build_front_end_router(
    inference_runner: InferenceRunner,
    local_worker_identity_query: LocalWorkerIdentityQuery,
    model_loader: ModelLoader,
) -> APIRouter:
    router = APIRouter(tags=["front-end"])

    @router.get(
        path=FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
        response_model=WorkerIdentityDto,
    )
    async def get_local_worker_identity() -> WorkerIdentityDto:
        worker_id = await local_worker_identity_query.get_local_worker_identity()
        return WorkerIdentityDto(service_id=worker_id.id)

    @router.post(
        path=FRONT_END_RUN_INFERENCE_PATH,
        response_model=RunInferenceResponse,
    )
    async def run_inference(
        request: RunInferenceRequest,
    ) -> RunInferenceResponse:
        result = await inference_runner.run_inference(
            flow_id=request.flow_id,
            bundle=request.bundle.to_ndarrays(),
        )
        return RunInferenceResponse(
            bundle=TensorBundleDto.from_ndarrays(result.bundle),
        )

    @router.post(path=FRONT_END_REGISTER_MODEL_PATH)
    async def register_model(request: RegisterModelRequest) -> None:
        await model_loader.register_model(request.model_dto)

    @router.post(
        path=FRONT_END_LOAD_MODEL_VARIANT_PATH,
        response_model=LoadModelVariantResponse,
    )
    async def load_model_variant(
        model_variant_json: Annotated[str, Form()],
        artifact: Annotated[UploadFile, File()],
    ) -> LoadModelVariantResponse:
        registration = StartModelVariantRegistrationRequest.model_validate_json(
            model_variant_json
        )

        async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "artifact.zip"
            async with aiofiles.open(archive_path, mode="wb") as archive_file:
                while chunk := await artifact.read(CHUNK_SIZE):
                    await archive_file.write(chunk)

            async with decompress_artifact_bundle(archive_path) as artifact_bundle:
                model_variant_id, artifact_ref = await model_loader.load_model_variant(
                    model_id=registration.model_id,
                    model_variant_info=registration.model_variant_info_dto,
                    artifact_bundle=artifact_bundle,
                )

        return LoadModelVariantResponse(
            model_variant_id=model_variant_id,
            artifact_ref=artifact_ref,
        )

    return router
