import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast, override

import boto3  # type: ignore[import-untyped]
from botocore import UNSIGNED  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from redis import asyncio as redis_asyncio

from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.s3.s3_artifact_store import S3ArtifactStore
from artifacts.storage.s3.s3_client import S3Client
from catalog.heartbeat.heartbeat_manager import HeartbeatManager
from catalog.redis.redis_directory import RedisDirectory
from catalog.redis.redis_registrar import RedisRegistrar
from integration.redis.model_partition.keys_factory import (
    StringModelPartitionKeyFactory,
)
from integration.redis.profiled_model_variant.keys_factory import (
    StringProfiledModelVariantKeyFactory,
)
from integration.redis.worker_profiles.execution_profile_keys import (
    StringExecutionProfileKeyFactory,
)
from integration.redis.worker_profiles.network_profile_keys import (
    StringNetworkProfileKeyFactory,
)
from integration.redis.worker_profiles.resource_profile_keys import (
    StringResourceProfileKeyFactory,
)
from integration.service_instance.worker_instance import WorkerInstance
from lifecycle.async_lifecycle import AsyncLifecycle
from worker.adapters.outbound.activity_manager.local_activity_manager import (
    LocalActivityManagerAdapter,
)
from worker.adapters.outbound.catalog.partition_deployment.redis_partition_deployment_catalog import (
    RedisPartitionDeploymentCatalog,
)
from worker.adapters.outbound.catalog.service.default_service_directory_gateway import (
    DefaultServiceDirectoryGateway,
)
from worker.adapters.outbound.contribution.in_memory_contribution_store import (
    InMemoryContributionStore,
)
from worker.adapters.outbound.hop_contribution.arrow.arrow_hop_invocation_contribution_sender import (
    ArrowHopInvocationContributionSender,
)
from worker.adapters.outbound.model_metadata.redis_model_partition_metadata_reader import (
    RedisModelPartitionMetadataReader,
)
from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor_factory import (
    OnnxPartitionExecutorFactory,
)
from worker.adapters.outbound.plan_store.in_memory_worker_plan_store import (
    InMemoryWorkerPlanStore,
)
from worker.adapters.outbound.profiling.model_execution.ort_engine_model_execution_profiler import (
    OrtModelExecutionProfiler,
)
from worker.adapters.outbound.profiling.model_execution.redis_model_execution_profile_store import (
    RedisModelExecutionProfileStore,
)
from worker.adapters.outbound.profiling.model_execution.redis_model_static_profile_reader import (
    RedisModelStaticProfileReader,
)
from worker.adapters.outbound.profiling.network.arrow.arrow_network_probe_client import (
    ArrowNetworkProbeClient,
)
from worker.adapters.outbound.profiling.network.publisher.redis_network_profile_publisher import (
    RedisNetworkProfilePublisher,
)
from worker.adapters.outbound.profiling.resource.cgroup_cpu_profiler import (
    CgroupCpuProfiler,
)
from worker.adapters.outbound.profiling.resource.cgroup_ram_profiler import (
    CgroupRamProfiler,
)
from worker.adapters.outbound.profiling.resource.no_gpu_profiler import NoGpuProfiler
from worker.adapters.outbound.profiling.resource.nvidia_gpu_profiler import (
    NvidiaGpuProfiler,
)
from worker.adapters.outbound.profiling.resource.publisher.redis_resource_profile_publisher import (
    RedisResourceProfilePublisher,
)
from worker.adapters.outbound.profiling.utils.nvidia_gpu_availability_checker import (
    NvidiaGPUAvailabilityChecker,
)
from worker.application.activity.default_activity_request_factory import (
    DefaultActivityRequestFactory,
)
from worker.application.deployment.default_worker_plan_deployer import (
    DefaultWorkerPlanDeployer,
)
from worker.application.hop_contribution_rcv.abc.contribution_gathering_policy import (
    HopContributionCollection,
    HopContributionCollectionKey,
)
from worker.application.hop_contribution_rcv.default_contribution_gathering_policy import (
    DefaultContributionGatheringPolicy,
)
from worker.application.hop_contribution_rcv.default_contribution_pipeline import (
    DefaultContributionPipeline,
)
from worker.application.hop_contribution_rcv.default_hop_invocation_contribution_receive_coordinator import (
    DefaultHopInvocationContributionReceiveCoordinator,
)
from worker.application.hop_contribution_snd.default_hop_contribution_propagator import (
    DefaultHopContributionPropagator,
)
from worker.application.hop_contribution_snd.default_hop_contribution_send_coordinator import (
    DefaultHopContributionDispatcher,
)
from worker.application.hop_contribution_snd.default_hop_output_router import (
    DefaultHopOutputRouter,
)
from worker.application.hop_handler.default_ingress_hop_handler import (
    DefaultIngressHopHandler,
)
from worker.application.hop_handler.default_partition_executor_registry import (
    DefaultPartitionExecutorRegistry,
)
from worker.application.hop_handler.default_partition_replica_hop_handler import (
    DefaultPartitionReplicaHopHandler,
)
from worker.application.hop_handler.default_sink_hop_handler import (
    DefaultSinkHopHandler,
)
from worker.application.hop_invocation.default_hop_invocation_request_assembler import (
    DefaultHopInvocationRequestAssembler,
)
from worker.application.hop_invocation.hop_invocation_coordinator import (
    HopInvocationCoordinator,
)
from worker.application.ports.outbound.profiling.resource.gpu_profiler import (
    GpuProfiler,
)
from worker.application.postbox.default_postbox import DefaultPostbox
from worker.application.profiling.default_model_execution_profiler import (
    DefaultModelExecutionProfiler,
)
from worker.application.profiling.default_model_execution_profiling_coordinator import (
    DefaultModelExecutionProfilingCoordinator,
)
from worker.application.profiling.default_network_profiling_coordinator import (
    DefaultNetworkProfilingCoordinator,
)
from worker.application.profiling.default_pending_models_poller import (
    DefaultPendingModelsPoller,
)
from worker.application.profiling.default_resource_profiling_coordinator import (
    DefaultResourceProfilingCoordinator,
)
from worker.application.scheduling.plan_based_partition_invocation_request_scheduler import (
    PlanBasedPartitionInvocationRequestScheduler,
)
from worker.application.sink.default_model_pass_result_retriever import (
    DefaultModelPassResultRetriever,
)
from worker.application.worker_identity.default_worker_identity_query import (
    DefaultWorkerIdentityQuery,
)
from worker.composition.composition_root import CompositionRoot, WorkerComposition
from worker.domain.activity.activity_request import ResourceType
from worker.domain.context.model_pass_context import ModelPassId
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)
from worker.domain.model.model_pass import ModelPassError, ModelPassResult
from worker.domain.profiling.model_execution.model_execution_profile import Backend


class DefaultCompositionRoot(CompositionRoot):
    @asynccontextmanager
    @override
    async def compose(
        self,
        service_instance: WorkerInstance,
    ) -> AsyncGenerator[WorkerComposition]:
        worker_id = service_instance.service_id

        async with AsyncExitStack() as stack:
            redis = await stack.enter_async_context(
                self.build_redis_connection_context()
            )
            await stack.enter_async_context(
                self.build_heartbeat_manager_context(redis, service_instance)
            )
            artifact_store = await stack.enter_async_context(
                self.build_artifact_store_context()
            )

            directory = RedisDirectory(redis)
            service_directory = DefaultServiceDirectoryGateway(directory)
            deployment_catalog = RedisPartitionDeploymentCatalog(redis)
            worker_plan_store = InMemoryWorkerPlanStore()
            activity_request_factory = DefaultActivityRequestFactory()
            activity_manager = LocalActivityManagerAdapter(
                {
                    ResourceType.COMPUTE: 1.0,
                    ResourceType.NETWORK: 1.0,
                }
            )

            partition_executor_registry = DefaultPartitionExecutorRegistry()
            worker_plan_deployer = DefaultWorkerPlanDeployer(
                worker_id=worker_id,
                worker_plan_store=worker_plan_store,
                artifact_store=artifact_store,
                model_partition_metadata_reader=RedisModelPartitionMetadataReader(
                    redis, StringModelPartitionKeyFactory
                ),
                partition_executor_factory=OnnxPartitionExecutorFactory(),
                partition_executor_registry=partition_executor_registry,
                partition_deployment_registrar=deployment_catalog,
            )

            contribution_dispatcher = DefaultHopContributionDispatcher(
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
                contribution_sender=ArrowHopInvocationContributionSender(
                    service_directory,
                    deployment_catalog,
                ),
            )

            request_assembler = DefaultHopInvocationRequestAssembler(worker_plan_store)

            hop_contribution_collection_postbox = DefaultPostbox[
                HopContributionCollectionKey,
                HopContributionCollection,
            ](
                key_compute_callable=lambda collection: collection.key,
            )
            contribution_pipeline = DefaultContributionPipeline(
                gathering_policy=DefaultContributionGatheringPolicy(worker_plan_store),
                contribution_store=InMemoryContributionStore[
                    HopContributionCollectionKey,
                    HopInvocationContribution,
                ](),
                outbox=hop_contribution_collection_postbox,
            )

            partition_replica_handler = DefaultPartitionReplicaHopHandler(
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
                hop_invocation_request_scheduler=PlanBasedPartitionInvocationRequestScheduler(
                    worker_plan_store
                ),
                partition_executor_registry=partition_executor_registry,
            )

            result_postbox = DefaultPostbox[ModelPassId, ModelPassResult](
                key_compute_callable=lambda result: result.context.model_pass_id
            )
            result_error_postbox = DefaultPostbox[ModelPassId, ModelPassError](
                key_compute_callable=lambda error: error.context.model_pass_id
            )
            model_pass_result_retriever = DefaultModelPassResultRetriever(
                result_inbox=result_postbox,
                error_inbox=result_error_postbox,
            )

            invocation_coordinator = HopInvocationCoordinator(
                max_concurrent_invocations=4,
                inbox=hop_contribution_collection_postbox,
                assembler=request_assembler,
                ingress_hop_handler=DefaultIngressHopHandler(),
                partition_replica_hop_handler=partition_replica_handler,
                sink_hop_handler=DefaultSinkHopHandler(result_postbox),
                contribution_propagator=DefaultHopContributionPropagator(
                    router=DefaultHopOutputRouter(worker_plan_store),
                    dispatcher=contribution_dispatcher,
                ),
            )

            await self.start_lifecycle(stack, partition_replica_handler)
            await self.start_lifecycle(stack, invocation_coordinator)
            await self.start_profiling_lifecycles(
                stack=stack,
                worker_instance=service_instance,
                redis=redis,
                artifact_store=artifact_store,
                service_directory=service_directory,
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
            )

            yield WorkerComposition(
                worker_plan_deployer=worker_plan_deployer,
                hop_contribution_receiver=DefaultHopInvocationContributionReceiveCoordinator(
                    contribution_pipeline
                ),
                model_pass_result_retriever=model_pass_result_retriever,
                worker_identity_query=DefaultWorkerIdentityQuery(worker_id),
            )

    async def start_profiling_lifecycles(
        self,
        stack: AsyncExitStack,
        worker_instance: WorkerInstance,
        redis: redis_asyncio.Redis,
        artifact_store: ArtifactStore,
        service_directory: DefaultServiceDirectoryGateway,
        activity_request_factory: DefaultActivityRequestFactory,
        activity_manager: LocalActivityManagerAdapter,
    ) -> None:
        worker_id = worker_instance.service_id

        gpu_profiler: GpuProfiler
        if NvidiaGPUAvailabilityChecker.check():
            gpu_profiler = NvidiaGpuProfiler()
        else:
            gpu_profiler = NoGpuProfiler()

        resource_coordinator = DefaultResourceProfilingCoordinator(
            worker_id=worker_id,
            gpu_profiler=gpu_profiler,
            cpu_profiler=CgroupCpuProfiler(),
            memory_profiler=CgroupRamProfiler(),
            res_profile_publisher=RedisResourceProfilePublisher(
                worker_id, redis, StringResourceProfileKeyFactory
            ),
        )
        network_coordinator = DefaultNetworkProfilingCoordinator(
            worker_id=worker_id,
            activity_request_factory=activity_request_factory,
            activity_manager=activity_manager,
            service_directory=service_directory,
            net_probe_client=ArrowNetworkProbeClient(),
            net_profile_publisher=RedisNetworkProfilePublisher(
                worker_id, redis, StringNetworkProfileKeyFactory
            ),
        )

        static_profile_reader = RedisModelStaticProfileReader(
            redis, StringProfiledModelVariantKeyFactory
        )
        execution_profile_store = RedisModelExecutionProfileStore(
            redis, StringExecutionProfileKeyFactory
        )
        pending_models_poller = DefaultPendingModelsPoller(
            worker_id=worker_id,
            model_static_profile_reader=static_profile_reader,
            model_execution_profile_reader=execution_profile_store,
        )

        backends = [Backend.CPU]
        if NvidiaGPUAvailabilityChecker.check():
            backends.append(Backend.GPU)

        model_execution_coordinator = DefaultModelExecutionProfilingCoordinator(
            worker_id=worker_id,
            pending_models_poller=pending_models_poller,
            model_static_profile_reader=static_profile_reader,
            model_execution_profiler=DefaultModelExecutionProfiler(
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
                backends=backends,
                engine_profiler=OrtModelExecutionProfiler(),
            ),
            artifact_store=artifact_store,
            model_exec_profile_publisher=execution_profile_store,
        )

        await self.start_lifecycle(stack, resource_coordinator)
        await self.start_lifecycle(stack, network_coordinator)
        await self.start_lifecycle(stack, pending_models_poller)
        await self.start_lifecycle(stack, model_execution_coordinator)

    @staticmethod
    async def start_lifecycle(
        stack: AsyncExitStack,
        lifecycle: AsyncLifecycle,
    ) -> None:
        stack.push_async_callback(lifecycle.stop)
        await lifecycle.start()

    @asynccontextmanager
    async def build_redis_connection_context(
        self,
    ) -> AsyncGenerator[redis_asyncio.Redis]:
        async with redis_asyncio.Redis(host="redis", port=6379) as redis:
            await redis.ping()
            yield redis

    @asynccontextmanager
    async def build_heartbeat_manager_context(
        self,
        redis: redis_asyncio.Redis,
        service_instance: WorkerInstance,
    ) -> AsyncGenerator[HeartbeatManager]:
        heartbeat_manager = HeartbeatManager(
            RedisRegistrar(redis, entry_ttl_s=60),
            service_instance,
            heartbeat_interval_s=20,
        )
        await heartbeat_manager.start()
        try:
            yield heartbeat_manager
        finally:
            await heartbeat_manager.stop()

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
                if error.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                    raise
                await asyncio.to_thread(client.create_bucket, Bucket=bucket_name)

            yield S3ArtifactStore(cast(S3Client, client), bucket_name)
        finally:
            client.close()
