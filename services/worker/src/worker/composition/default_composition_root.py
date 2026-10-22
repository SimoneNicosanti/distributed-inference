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
from integration.service_instance.service_instance import CapabilityType
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
from worker.adapters.outbound.model_metadata.redis_model_partition_metadata_reader import (
    RedisModelPartitionMetadataReader,
)
from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor_factory import (
    OnnxPartitionExecutorFactory,
)
from worker.adapters.outbound.partition_invocation.sender.arrow.arrow_model_pass_result_contribution_sender import (
    ArrowModelPassResultContributionSender,
)
from worker.adapters.outbound.partition_invocation.sender.arrow.arrow_partition_invocation_contribution_sender import (
    ArrowPartitionInvocationContributionSender,
)
from worker.adapters.outbound.partition_invocation.store.in_memory_partition_invocation_contribution_store import (
    InMemoryPartitionInvocationContributionStore,
)
from worker.adapters.outbound.plan_store.in_memory_ingress_plan_store import (
    InMemoryIngressPlanStore,
)
from worker.adapters.outbound.plan_store.in_memory_worker_plan_store import (
    InMemoryPlanStore,
)
from worker.adapters.outbound.profiling.model_execution.onnx_model_execution_profiler import (
    OnnxModelExecutionProfiler,
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
from worker.adapters.outbound.profiling.resource.publisher.redis_resource_profile_publisher import (
    RedisResourceProfilePublisher,
)
from worker.application.activity.default_activity_request_factory import (
    DefaultActivityRequestFactory,
)
from worker.application.deployment.default_ingress_plan_deployer import (
    DefaultIngressPlanDeployer,
)
from worker.application.deployment.default_worker_plan_deployer import (
    DefaultWorkerPlanDeployer,
)
from worker.application.ingress.default_ingress_input_router import (
    DefaultIngressInputRouter,
)
from worker.application.ingress.default_model_invocation_coordinator import (
    DefaultModelInvocationCoordinator,
)
from worker.application.ingress.default_model_pass_coordinator import (
    DefaultModelPassCoordinator,
)
from worker.application.ingress.default_result_gatherer import DefaultResultGatherer
from worker.application.partition_execution.default_partition_execution_coordinator import (
    DefaultPartitionExecutionCoordinator,
)
from worker.application.partition_execution.default_partition_executor_registry import (
    DefaultPartitionExecutorRegistry,
)
from worker.application.partition_invocation.default_partition_invocation_coordinator import (
    DefaultPartitionInvocationCoordinator,
)
from worker.application.partition_invocation.input.default_partition_invocation_contribution_inbox import (
    DefaultPartitionInvocationContributionInbox,
)
from worker.application.partition_invocation.input.default_partition_invocation_request_assembler import (
    DefaultPartitionInvocationRequestAssembler,
)
from worker.application.partition_invocation.output.default_contribution_send_coordinator import (
    DefaultContributionSendCoordinator,
)
from worker.application.partition_invocation.output.default_partition_output_router import (
    DefaultPartitionOutputRouter,
)
from worker.application.profiling.default_model_execution_profiling_coordinator import (
    DefaultModelExecutionProfileCoordinator,
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
from worker.application.worker_identity.default_worker_identity_query import (
    DefaultWorkerIdentityQuery,
)
from worker.composition.composition_root import CompositionRoot, WorkerComposition
from worker.domain.activity.activity_request import ResourceType


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
            worker_plan_store = InMemoryPlanStore()
            ingress_plan_store = InMemoryIngressPlanStore()
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
                    redis
                ),
                partition_executor_factory=OnnxPartitionExecutorFactory(),
                partition_executor_registry=partition_executor_registry,
                partition_deployment_registrar=deployment_catalog,
            )

            contribution_send_coordinator = DefaultContributionSendCoordinator(
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
                partition_invocation_contribution_sender=ArrowPartitionInvocationContributionSender(
                    service_directory,
                    deployment_catalog,
                ),
                model_pass_result_contribution_sender=ArrowModelPassResultContributionSender(
                    service_directory
                ),
            )

            request_assembler = DefaultPartitionInvocationRequestAssembler(
                worker_plan_store
            )
            contribution_inbox = DefaultPartitionInvocationContributionInbox(
                routing_plan_reader=worker_plan_store,
                contribution_store=InMemoryPartitionInvocationContributionStore(),
                partition_invocation_request_assembler=request_assembler,
            )
            execution_coordinator = DefaultPartitionExecutionCoordinator(
                activity_request_factory=activity_request_factory,
                activity_manager=activity_manager,
                partition_invocation_request_scheduler=PlanBasedPartitionInvocationRequestScheduler(
                    worker_plan_store
                ),
                partition_executor_registry=partition_executor_registry,
            )
            invocation_coordinator = DefaultPartitionInvocationCoordinator(
                max_concurrent_invocations=4,
                contribution_inbox=contribution_inbox,
                partition_execution_coordinator=execution_coordinator,
                output_router=DefaultPartitionOutputRouter(worker_plan_store),
                contribution_sender_coordinator=contribution_send_coordinator,
            )

            await self.start_lifecycle(stack, execution_coordinator)
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

            ingress_plan_deployer = None
            model_invocation_coordinator = None
            result_contribution_sink = None

            if service_instance.get_capabilities_by_type(CapabilityType.RESULT_SINKER):
                ingress_plan_deployer = DefaultIngressPlanDeployer(ingress_plan_store)
                result_gatherer = DefaultResultGatherer()
                model_pass_coordinator = DefaultModelPassCoordinator(
                    input_router=DefaultIngressInputRouter(),
                    ingress_plan_reader=ingress_plan_store,
                    contribution_sender_coordinator=contribution_send_coordinator,
                    result_waiter=result_gatherer,
                )
                model_invocation_coordinator = DefaultModelInvocationCoordinator(
                    result_sink_id=worker_id,
                    active_plan_version_reader=ingress_plan_store,
                    model_pass_coordinator=model_pass_coordinator,
                )
                result_contribution_sink = result_gatherer

            yield WorkerComposition(
                worker_plan_deployer=worker_plan_deployer,
                partition_invocation_coordinator=invocation_coordinator,
                worker_identity_query=DefaultWorkerIdentityQuery(worker_id),
                ingress_plan_deployer=ingress_plan_deployer,
                model_invocation_coordinator=model_invocation_coordinator,
                result_contribution_sink=result_contribution_sink,
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

        resource_coordinator = DefaultResourceProfilingCoordinator(
            worker_id=worker_id,
            gpu_profiler=NoGpuProfiler(),
            cpu_profiler=CgroupCpuProfiler(),
            memory_profiler=CgroupRamProfiler(),
            res_profile_publisher=RedisResourceProfilePublisher(worker_id, redis),
        )
        network_coordinator = DefaultNetworkProfilingCoordinator(
            worker_id=worker_id,
            activity_request_factory=activity_request_factory,
            activity_manager=activity_manager,
            service_directory=service_directory,
            net_probe_client=ArrowNetworkProbeClient(),
            net_profile_publisher=RedisNetworkProfilePublisher(worker_id, redis),
        )

        static_profile_reader = RedisModelStaticProfileReader(redis)
        execution_profile_store = RedisModelExecutionProfileStore(redis)
        pending_models_poller = DefaultPendingModelsPoller(
            worker_id=worker_id,
            model_static_profile_reader=static_profile_reader,
            model_execution_profile_reader=execution_profile_store,
        )
        model_execution_coordinator = DefaultModelExecutionProfileCoordinator(
            worker_id=worker_id,
            pending_models_poller=pending_models_poller,
            model_static_profile_reader=static_profile_reader,
            activity_request_factory=activity_request_factory,
            activity_manager=activity_manager,
            model_execution_profiler=OnnxModelExecutionProfiler(),
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
