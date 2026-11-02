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
from redis import asyncio as redis_asyncio

from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.s3.s3_artifact_store import S3ArtifactStore
from artifacts.storage.s3.s3_client import S3Client
from catalog.redis.hop.keys_factory import StringHopDeploymentKeyFactory
from catalog.redis.hop.redis_hop_deployment_catalog import RedisHopDeploymentCatalog
from catalog.redis.service.redis_service_catalog import RedisServiceCatalog
from front_end.adapters.outbound.inference.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from front_end.adapters.outbound.inference.arrow_worker_sink_gateway import (
    ArrowWorkerSinkGateway,
)
from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from front_end.adapters.outbound.loading.artifact_store_model_artifact_uploader import (
    ArtifactStoreModelArtifactUploader,
)
from front_end.adapters.outbound.loading.http_model_manager_gateway import (
    HttpModelManagerGateway,
)
from front_end.adapters.outbound.optimization.http_local_worker_identity_gateway import (
    HttpLocalWorkerIdentityGateway,
)
from front_end.adapters.outbound.optimization.http_optimization_gateway import (
    HttpOptimizationGateway,
)
from front_end.application.inference.default_inference_runner import (
    DefaultInferenceRunner,
)
from front_end.application.loading.default_model_loader import DefaultModelLoader
from front_end.composition.composition_root import (
    CompositionRoot,
    FrontEndComposition,
)
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.worker.http_worker_constants import (
    DEFAULT_LOCAL_WORKER_HTTP_BASE_URL,
)

DEFAULT_MODEL_MANAGER_HTTP_BASE_URL = "http://model-manager:8000"
DEFAULT_CONTROLLER_HTTP_BASE_URL = "http://controller:8000"
DEFAULT_REDIS_URL = "redis://redis:6379/0"


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
            redis = await stack.enter_async_context(
                self.build_redis_connection_context()
            )
            directory = RedisServiceCatalog(redis, entry_ttl_s=60)

            inference = DefaultInferenceRunner(
                flow_plan_reader=RedisFlowPlanReader(redis, StringFlowPlanKeyFactory()),
                hop_deployment_directory=RedisHopDeploymentCatalog(
                    redis, StringHopDeploymentKeyFactory
                ),
                ingress_gateway=ArrowWorkerIngressGateway(
                    directory,
                    timeout_s=float(
                        os.environ.get("WORKER_HOP_SUBMIT_TIMEOUT_S", "30")
                    ),
                ),
                sink_gateway=ArrowWorkerSinkGateway(
                    directory,
                    timeout_s=float(os.environ.get("WORKER_RESULT_TIMEOUT_S", "300")),
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
                worker_identity_gateway=HttpLocalWorkerIdentityGateway(
                    http_client,
                    os.environ.get(
                        "WORKER_HTTP_BASE_URL", DEFAULT_LOCAL_WORKER_HTTP_BASE_URL
                    ),
                ),
                model_loader=model_loader,
                optimization_gateway=HttpOptimizationGateway(
                    http_client,
                    os.environ.get(
                        "CONTROLLER_HTTP_BASE_URL", DEFAULT_CONTROLLER_HTTP_BASE_URL
                    ),
                    timeout_s=float(
                        os.environ.get("CONTROLLER_OPTIMIZATION_TIMEOUT_S", "300")
                    ),
                ),
            )

    @asynccontextmanager
    async def build_redis_connection_context(
        self,
    ) -> AsyncGenerator[redis_asyncio.Redis]:
        redis = redis_asyncio.Redis.from_url(
            os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        )
        try:
            yield redis
        finally:
            await redis.aclose()

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
