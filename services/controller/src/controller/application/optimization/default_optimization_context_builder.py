from typing import override

from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from controller.application.ports.outbound.profile.execution_profile_store_gateway import (
    ExecutionProfileStoreGateway,
)
from controller.application.ports.outbound.profile.network_profile_store_gateway import (
    NetworkProfileStoreGateway,
)
from controller.application.ports.outbound.profile.resource_profile_store_gateway import (
    ResourceProfileStoreGateway,
)
from controller.application.ports.outbound.profile.static_profile_store_gateway import (
    StaticProfileStoreGateway,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.profile.network_topology import NetworkTopology


class DefaultOptimizationContextBuilder(OptimizationContextBuilder):
    def __init__(
        self,
        worker_registry_gateway: WorkerDirectoryGateway,
        static_profile_store_gateway: StaticProfileStoreGateway,
        network_profile_store_gateway: NetworkProfileStoreGateway,
        execution_profile_store_gateway: ExecutionProfileStoreGateway,
        resource_profile_store_gateway: ResourceProfileStoreGateway,
    ) -> None:
        self._worker_registry_gateway = worker_registry_gateway
        self._static_profile_store_gateway = static_profile_store_gateway
        self._network_profile_store_gateway = network_profile_store_gateway
        self._execution_profile_store_gateway = execution_profile_store_gateway
        self._resource_profile_store_gateway = resource_profile_store_gateway

    @override
    async def build_optimization_context(self) -> OptimizationContext:
        all_static_profiles = await (
            self._static_profile_store_gateway.get_all_model_variant_static_profiles()
        )
        static_profiles_by_variant = {
            profile.model_variant_id: profile for profile in all_static_profiles
        }
        if len(static_profiles_by_variant) != len(all_static_profiles):
            raise ValueError("Duplicate static profiles for a model variant")

        active_workers = await self._worker_registry_gateway.get_registered_worker_ids()

        network_connections = await (
            self._network_profile_store_gateway.get_network_connections_by_workers(
                active_workers
            )
        )
        network_topology = NetworkTopology(
            workers=active_workers,
            connections=network_connections,
        )

        resource_profiles = await (
            self._resource_profile_store_gateway.get_resource_profiles_for_workers(
                active_workers
            )
        )
        resource_profiles_by_worker = {
            profile.worker_id: profile for profile in resource_profiles
        }
        if len(resource_profiles_by_worker) != len(resource_profiles):
            raise ValueError("Duplicate resource profiles for a worker")

        workers_execution_profiles = await self._execution_profile_store_gateway.get_execution_profiles_for_workers_and_variants(
            active_workers,
            static_profiles_by_variant.keys(),
        )
        execution_profiles_by_worker = {
            worker_profiles.worker_id: worker_profiles
            for worker_profiles in workers_execution_profiles
        }
        if len(execution_profiles_by_worker) != len(workers_execution_profiles):
            raise ValueError("Duplicate execution profiles for a worker")

        return OptimizationContext(
            static_profiles_by_variant=static_profiles_by_variant,
            network_topology=network_topology,
            resource_profiles_by_worker=resource_profiles_by_worker,
            execution_profiles_by_worker=execution_profiles_by_worker,
        )
