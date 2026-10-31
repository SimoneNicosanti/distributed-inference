import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast, override

import boto3
import httpx
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.s3.s3_artifact_store import S3ArtifactStore
from artifacts.storage.s3.s3_client import S3Client
from front_end.adapters.outbound.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from front_end.adapters.outbound.artifact_store_model_artifact_uploader import (
    ArtifactStoreModelArtifactUploader,
)
from front_end.adapters.outbound.http_model_manager_gateway import (
    HttpModelManagerGateway,
)
from front_end.adapters.outbound.http_worker_identity_gateway import (
    HttpWorkerIdentityGateway,
)
from front_end.application.inference.default_inference_runner import (
    DefaultInferenceRunner,
)
from front_end.application.loading.default_model_loader import DefaultModelLoader
from front_end.composition.composition_root import (
    CompositionRoot,
    FrontEndComposition,
)
from integration.worker.http_worker_constants import (
    DEFAULT_LOCAL_WORKER_HTTP_BASE_URL,
)

DEFAULT_MODEL_MANAGER_HTTP_BASE_URL = "http://model-manager:8000"
DEFAULT_LOCAL_WORKER_MODEL_INVOCATION_LOCATION = "grpc+tcp://127.0.0.1:8818"


class DefaultCompositionRoot(CompositionRoot):
    @asynccontextmanager
    @override
    async def compose(self) -> AsyncGenerator[FrontEndComposition]:
        async with AsyncExitStack() as stack:
            http_client = await stack.enter_async_context(
                self.build_http_client_context()
            )
            artifact_store = await stack.enter_async_context(
                self.build_artifact_store_context()
            )

            inference = DefaultInferenceRunner(
                ingress_gateway=ArrowWorkerIngressGateway(
                    os.environ.get(
                        "WORKER_MODEL_INVOCATION_LOCATION",
                        DEFAULT_LOCAL_WORKER_MODEL_INVOCATION_LOCATION,
                    )
                ),
                worker_identity_gateway=HttpWorkerIdentityGateway(
                    http_client,
                    os.environ.get(
                        "WORKER_HTTP_BASE_URL",
                        DEFAULT_LOCAL_WORKER_HTTP_BASE_URL,
                    ),
                ),
            )

            model_loader = DefaultModelLoader(
                model_manager_gateway=HttpModelManagerGateway(
                    http_client,
                    os.environ.get(
                        "MODEL_MANAGER_HTTP_BASE_URL",
                        DEFAULT_MODEL_MANAGER_HTTP_BASE_URL,
                    ),
                ),
                model_artifact_uploader=ArtifactStoreModelArtifactUploader(
                    artifact_store
                ),
            )

            yield FrontEndComposition(
                inference_runner=inference,
                local_worker_identity_query=inference,
                model_loader=model_loader,
            )

    @asynccontextmanager
    async def build_http_client_context(self) -> AsyncGenerator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        ) as http_client:
            yield http_client

    @asynccontextmanager
    async def build_artifact_store_context(self) -> AsyncGenerator[ArtifactStore]:
        bucket_name = os.environ.get("SEAWEEDFS_S3_BUCKET", "artifacts")

        client = boto3.client(
            "s3",
            endpoint_url="http://seaweedfs:8333",
            config=Config(signature_version=UNSIGNED),
        )

        try:
            try:
                await asyncio.to_thread(client.head_bucket, Bucket=bucket_name)
            except ClientError as error:
                if error.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    await asyncio.to_thread(client.create_bucket, Bucket=bucket_name)
                else:
                    raise

            yield S3ArtifactStore(cast(S3Client, client), bucket_name)
        finally:
            client.close()
