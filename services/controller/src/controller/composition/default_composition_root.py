from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import override

import httpx
from redis import asyncio as redis_asyncio

from advertisement.advertising_manager import AdvertisingManager
from advertisement.http.http_capability_life_checker import (
    build_http_capability_life_checkers,
)
from catalog.api.service.service_directory import ServiceDirectory
from catalog.redis.service.redis_service_directory import RedisServiceDirectory
from catalog.redis.service.redis_service_registrar import RedisServiceRegistrar
from catalog.view.worker.default_worker_directory import DefaultWorkerDirectory
from controller.adapters.outbound.directory.default_worker_directory_gateway import (
    DefaultWorkerDirectoryGateway,
)
from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.compact_variables import (
    CompactVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.gurobi_compact_formulation_compiler import (
    GurobiCompactFormulationCompiler,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.gurobi_formulation_model_builder import (
    GurobiFormulationModelBuilder,
)
from controller.adapters.outbound.optimization.gurobi.result_translation.formulations.compact.gurobi_compact_solved_formulation_model_translator import (
    GurobiCompactSolvedFormulationModelTranslator,
)
from controller.adapters.outbound.optimization.gurobi.solver.gurobi_solver import (
    GurobiSolver,
)
from controller.adapters.outbound.optimization.gurobi.solver_config_builder.gurobi_solver_config_builder import (
    GurobiSolverConfigBuilder,
)
from controller.adapters.outbound.partition_builder.http_partition_builder_gateway import (
    HttpPartitionBuilderGateway,
)
from controller.adapters.outbound.plan_share.http_plan_share_gateway import (
    HttpPlanShareGateway,
)
from controller.adapters.outbound.plan_version.redis_plan_version_factory import (
    RedisPlanVersionFactory,
)
from controller.adapters.outbound.profile.redis_execution_profile_store_gateway import (
    RedisExecutionProfileStoreGateway,
)
from controller.adapters.outbound.profile.redis_network_profile_store_gateway import (
    RedisNetworkProfileStoreGateway,
)
from controller.adapters.outbound.profile.redis_resource_profile_store_gateway import (
    RedisResourceProfileStoreGateway,
)
from controller.adapters.outbound.profile.redis_static_profile_store_gateway import (
    RedisStaticProfileStoreGateway,
)
from controller.adapters.outbound.store.in_memory_optimization_solution_store import (
    InMemoryOptimizationSolutionStore,
)
from controller.adapters.outbound.store.redis_flow_plan_publisher import (
    RedisFlowPlanPublisher,
)
from controller.application.deployment.abc.actuation_manager import ActuationManager
from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.default_actuation_manager import (
    DefaultActuationManager,
)
from controller.application.deployment.default_model_partition_build_manager import (
    DefaultModelPartitionBuildManager,
)
from controller.application.deployment.default_plan_mappings_builder import (
    DefaultPlanMappingsBuilder,
)
from controller.application.deployment.default_plan_three_phase_dispatcher import (
    DefaultPlanThreePhaseDispatcher,
)
from controller.application.deployment.default_solution_to_plan_converter import (
    DefaultSolutionToPlanConverter,
)
from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.optimization.abc.solver_strategy import SolverStrategy
from controller.application.optimization.default_optimization_context_builder import (
    DefaultOptimizationContextBuilder,
)
from controller.application.optimization.default_optimization_manager import (
    DefaultOptimizationManager,
)
from controller.application.optimization.default_optimization_solution_enricher import (
    DefaultOptimizationSolutionEnricher,
)
from controller.application.optimization.formulation.compact.compact_formulation_data_builder import (
    CompactFormulationDataBuilder,
)
from controller.application.optimization.strategies.full_solver_strategy import (
    FullSolverStrategy,
)
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from controller.application.ports.outbound.store.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.composition.composition_root import CompositionRoot
from controller.domain.optimization.solving.formulation.compact.compact_formulation_config import (
    CompactFormulationConfig,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.redis.model.keys_factory import StringModelKeyFactory
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
from integration.service_instance.service_instance import ServiceInstance


class DefaultCompositionRoot(CompositionRoot):
    def __init__(self, solver_config: GurobiSolverConfig | None = None) -> None:
        self._solver_config = solver_config

    @asynccontextmanager
    @override
    async def compose(
        self,
        service_instance: ServiceInstance,
    ) -> AsyncGenerator[OptimizationManager]:

        async with AsyncExitStack() as stack:
            solution_store = InMemoryOptimizationSolutionStore()

            redis = await stack.enter_async_context(
                self.build_redis_connection_context()
            )

            await stack.enter_async_context(
                self.build_advertising_manager_context(redis, service_instance)
            )

            directory = RedisServiceDirectory(redis)
            worker_directory = DefaultWorkerDirectory(directory)
            worker_directory_gateway = DefaultWorkerDirectoryGateway(worker_directory)

            optimization_manager = self.build_optimization_manager(
                solution_store, redis, worker_directory_gateway
            )

            await stack.enter_async_context(
                self.build_actuation_manager_context(
                    solution_store, directory, worker_directory_gateway, redis
                )
            )

            yield optimization_manager

    def build_optimization_manager(
        self,
        solution_store: OptimizationSolutionStore,
        redis: redis_asyncio.Redis,
        worker_directory_gateway: WorkerDirectoryGateway,
    ) -> OptimizationManager:

        optimization_context_builder = self.build_optimization_context_builder(
            redis, worker_directory_gateway
        )
        solver_config = self._solver_config or GurobiSolverConfig(
            output_flag=True,
            time_limit_s=180,
        )

        formulation_config = CompactFormulationConfig(
            max_components=5,
            priority_count=3,
            compute_utilization_limit=0.8,
            network_utilization_limit=0.8,
            relative_error=0.1,
            objective=CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        )

        solver_strategy: SolverStrategy = FullSolverStrategy(
            formulation_data_builder=CompactFormulationDataBuilder(),
            formulation_model_builder=GurobiFormulationModelBuilder(
                GurobiCompactFormulationCompiler(),
                non_convex=0,
                output_flag=solver_config.output_flag,
            ),
            solver_config_builder=GurobiSolverConfigBuilder(
                output_flag=solver_config.output_flag,
                time_limit_s=solver_config.time_limit_s,
                mip_gap=solver_config.mip_gap,
                threads=solver_config.threads,
                seed=solver_config.seed,
                observer=solver_config.observer,
                record_incumbent_history=solver_config.record_incumbent_history,
            ),
            solver=GurobiSolver[CompactFormulationData, CompactVariables](),
            solved_model_translator=GurobiCompactSolvedFormulationModelTranslator(),
            formulation_config=formulation_config,
        )
        optimization_manager = DefaultOptimizationManager(
            optimization_context_builder,
            solver_strategy,
            DefaultOptimizationSolutionEnricher(),
            solution_store,
        )

        return optimization_manager

    @asynccontextmanager
    async def build_actuation_manager_context(
        self,
        solution_observer: SolutionObserver,
        directory: ServiceDirectory,
        worker_directory_gateway: WorkerDirectoryGateway,
        redis: redis_asyncio.Redis,
    ) -> AsyncGenerator[ActuationManager]:

        async with AsyncExitStack() as stack:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=5.0),
                    limits=httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                    ),
                )
            )

            partition_builder_gateway = HttpPartitionBuilderGateway(
                http_client,
                directory,
            )
            plan_share_gateway = HttpPlanShareGateway(
                http_client,
                worker_directory_gateway,
            )

            plan_version_factory = RedisPlanVersionFactory(redis)
            service_plan_builder = DefaultSolutionToPlanConverter()

            actuation_manager = DefaultActuationManager(
                solution_observer,
                DefaultModelPartitionBuildManager(partition_builder_gateway),
                DefaultPlanMappingsBuilder(plan_version_factory, service_plan_builder),
                DefaultPlanThreePhaseDispatcher(
                    plan_share_gateway,
                    RedisFlowPlanPublisher(redis, StringFlowPlanKeyFactory()),
                ),
            )

            stack.push_async_callback(actuation_manager.stop)
            await actuation_manager.start()

            yield actuation_manager

    def build_optimization_context_builder(
        self,
        redis: redis_asyncio.Redis,
        worker_directory_gateway: WorkerDirectoryGateway,
    ) -> OptimizationContextBuilder:

        static_profile_store_gateway = RedisStaticProfileStoreGateway(
            redis, StringModelKeyFactory, StringProfiledModelVariantKeyFactory
        )
        network_profile_store_gateway = RedisNetworkProfileStoreGateway(
            redis, StringNetworkProfileKeyFactory
        )
        execution_profile_store_gateway = RedisExecutionProfileStoreGateway(
            redis, StringExecutionProfileKeyFactory
        )
        resource_profile_store_gateway = RedisResourceProfileStoreGateway(
            redis, StringResourceProfileKeyFactory
        )

        optimization_context_builder = DefaultOptimizationContextBuilder(
            worker_directory_gateway,
            static_profile_store_gateway,
            network_profile_store_gateway,
            execution_profile_store_gateway,
            resource_profile_store_gateway,
        )

        return optimization_context_builder

    @asynccontextmanager
    async def build_redis_connection_context(
        self,
    ) -> AsyncGenerator[redis_asyncio.Redis]:
        async with redis_asyncio.Redis(host="redis", port=6379) as redis:
            await redis.ping()
            yield redis

    @asynccontextmanager
    async def build_advertising_manager_context(
        self, redis: redis_asyncio.Redis, service_instance: ServiceInstance
    ) -> AsyncGenerator[AdvertisingManager]:
        async with httpx.AsyncClient() as http_client:
            registrar = RedisServiceRegistrar(redis, entry_ttl_s=60)
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
