from collections.abc import Iterator

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from distributed_inference.adapters.inbound.model_manager.http.schema import (
    DownloadSubModelRequest,
    GenerateSubModelRequest,
    GenerateSubModelResponse,
    GetModelGraphRequest,
    GetModelGraphResponse,
    RegisterModelRequest,
    RegisterModelResponse,
    UploadModelVersionResponse,
)
from distributed_inference.application.model_manager.contracts.model_manager import (
    ModelManager,
)
from distributed_inference.domain.identifiers import ModelId
from distributed_inference.domain.model_graph_info import ModelInfo


def build_model_manager_router(
    model_manager: ModelManager,
) -> APIRouter:

    router = APIRouter(
        prefix="/model-manager",
        tags=["model-manager"],
    )

    @router.post(
        "/models",
        response_model=ModelId,
    )
    def register_model(
        request: RegisterModelRequest,
    ) -> RegisterModelResponse:
        model_id = model_manager.register_model(
            owner_id=request.owner_id,
            model_name=request.model_name,
        )

        return RegisterModelResponse(model_id=model_id)

    @router.post(
        "/model-versions",
        response_model=UploadModelVersionResponse,
    )
    def upload_model_version(
        model_id_json: str = Form(),
        model_info_json: str = Form(),
        artifact: UploadFile = File(),
    ) -> UploadModelVersionResponse:
        model_id = ModelId.model_validate_json(model_id_json)
        model_info = ModelInfo.model_validate_json(model_info_json)

        model_version_id = model_manager.upload_model_version(
            model_id=model_id,
            model_info=model_info,
            binary_io=artifact.file,
        )

        return UploadModelVersionResponse(
            model_version_id=model_version_id,
        )

    @router.post(
        "/sub-models",
        response_model=GenerateSubModelResponse,
    )
    def generate_sub_model(
        request: GenerateSubModelRequest,
    ) -> GenerateSubModelResponse:
        sub_model_id = model_manager.generate_sub_model(
            model_version_id=request.model_version_id,
            layers=request.layers,
        )

        return GenerateSubModelResponse(
            sub_model_id=sub_model_id,
        )

    @router.post("/sub-models/download")
    def download_sub_model(
        request: DownloadSubModelRequest,
    ) -> StreamingResponse:
        def stream_artifact() -> Iterator[bytes]:
            with model_manager.download_sub_model(request.sub_model_id) as binary_io:
                while chunk := binary_io.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            stream_artifact(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": ('attachment; filename="sub_model.onnx"')},
        )

    @router.get("/model-versions/graph", response_model=GetModelGraphResponse)
    def get_model_graph(
        request: GetModelGraphRequest,
    ) -> GetModelGraphResponse:
        model_graph = model_manager.get_model_graph(request.model_version_id)

        return GetModelGraphResponse(model_graph=model_graph)

    ## TODO ADD CHECK EXISTANCE
    return router
