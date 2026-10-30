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

from advertisement.advertising_manager import AdvertisingManager
from advertisement.http.http_capability_life_checker import (
    build_http_capability_life_checkers,
)
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.s3.s3_artifact_store import S3ArtifactStore
from artifacts.storage.s3.s3_client import S3Client
from catalog.api.registrar import Registrar
from catalog.redis.redis_registrar import RedisRegistrar
from integration.redis.model.keys_factory import StringModelKeyFactory
from integration.redis.model_partition.keys_factory import (
    StringModelPartitionKeyFactory,
)
from integration.redis.profiled_model_variant.keys_factory import (
    StringProfiledModelVariantKeyFactory,
)
from integration.service_instance.service_instance import ServiceInstance
from model_manager.adapters.outbound.model_metadata_store.redis_model_metadata_store import (
    RedisModelMetadataStore,
)
from model_manager.adapters.outbound.model_optimizer.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from model_manager.adapters.outbound.model_partition_extractor.onnx_model_partition_extractor import (
    OnnxModelPartitionExtractor,
)
from model_manager.adapters.outbound.model_profiler.onnx_contraction_computer import (
    OnnxContractionComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_invariant_properties_computer import (
    OnnxInvariantPropertiesComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_model_topology_builder import (
    OnnxModelTopologyBuilder,
)
from model_manager.adapters.outbound.model_profiler.onnx_shape_profile_computer import (
    OnnxShapeProfileComputer,
)
from model_manager.application.default_artifact_ref_factory import (
    DefaultArtifactRefFactory,
)
from model_manager.application.default_model_manager import DefaultModelManager
from model_manager.application.default_model_profiler import DefaultModelProfiler
from model_manager.application.default_model_variant_id_factory import (
    DefaultModelVariantIdFactory,
)
from model_manager.application.ports.inbound.model_partitioner import ModelPartitioner
from model_manager.application.ports.inbound.model_registry import ModelRegistry
from model_manager.application.ports.inbound.model_variant_registry import (
    ModelVariantRegistry,
)
from model_manager.application.ports.outbound.model_partition_extractor import (
    ModelPartitionExtractor,
)
from model_manager.application.ports.outbound.model_profiler import ModelProfiler
from model_manager.composition.composition_root import CompositionRoot


class DefaultCompositionRoot(CompositionRoot):
    @asynccontextmanager
    @override
    async def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[tuple[ModelRegistry, ModelVariantRegistry, ModelPartitioner]]:

        async with AsyncExitStack() as stack:
            redis = await stack.enter_async_context(
                self.build_redis_connection_context()
            )

            registrar = RedisRegistrar(redis, entry_ttl_s=60)
            await stack.enter_async_context(
                self.build_advertising_manager_context(registrar, service_instance)
            )

            model_profiler = self.build_model_profiler()

            artifact_store_cm = self.build_artifact_store()
            artifact_store = await stack.enter_async_context(artifact_store_cm)
            artifact_ref_factory = DefaultArtifactRefFactory()

            model_metadata_store = RedisModelMetadataStore(
                redis,
                StringModelKeyFactory,
                StringProfiledModelVariantKeyFactory,
                StringModelPartitionKeyFactory,
            )

            model_splitter = self.build_model_partition_extractor()

            model_manager = DefaultModelManager(
                model_profiler,
                artifact_store,
                DefaultModelVariantIdFactory(),
                artifact_ref_factory,
                model_metadata_store,
                model_splitter,
            )

            yield model_manager, model_manager, model_manager

    @asynccontextmanager
    async def build_redis_connection_context(
        self,
    ) -> AsyncGenerator[redis_asyncio.Redis]:
        async with redis_asyncio.Redis(host="redis", port=6379) as redis:
            await redis.ping()
            yield redis

    @asynccontextmanager
    async def build_advertising_manager_context(
        self,
        registrar: Registrar,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[AdvertisingManager]:
        async with httpx.AsyncClient() as http_client:
            advertising_manager = AdvertisingManager(
                registrar,
                service_instance.service_id,
                build_http_capability_life_checkers(
                    service_instance,
                    http_client,
                ),
                heartbeat_interval_s=20,
            )
            await advertising_manager.start()
            try:
                yield advertising_manager
            finally:
                await advertising_manager.stop()

    @staticmethod
    def build_model_profiler() -> ModelProfiler:

        model_topology_builder = OnnxModelTopologyBuilder()
        invariant_properties_computer = OnnxInvariantPropertiesComputer()
        shape_profile_computer = OnnxShapeProfileComputer()

        onnx_model_optimizer = OnnxModelOptimizer()

        contraction_computer = OnnxContractionComputer(
            onnx_model_optimizer,
            model_topology_builder,
        )

        model_profiler = DefaultModelProfiler(
            model_topology_builder,
            invariant_properties_computer,
            shape_profile_computer,
            contraction_computer,
        )

        return model_profiler

    @staticmethod
    def build_model_partition_extractor() -> ModelPartitionExtractor:
        return OnnxModelPartitionExtractor()

    @asynccontextmanager
    async def build_artifact_store(self) -> AsyncGenerator[ArtifactStore]:
        bucket_name = os.environ.get("SEAWEEDFS_S3_BUCKET", "artifacts")

        client = boto3.client(
            "s3",
            endpoint_url="http://seaweedfs:8333",
            config=Config(signature_version=UNSIGNED),
        )

        try:
            try:
                await asyncio.to_thread(
                    client.head_bucket,
                    Bucket=bucket_name,
                )
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    await asyncio.to_thread(
                        client.create_bucket,
                        Bucket=bucket_name,
                    )
                else:
                    raise

            yield S3ArtifactStore(cast(S3Client, client), bucket_name)
        finally:
            client.close()
