import asyncio
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.identifiers.identifiers import WorkerId
from worker.application.ports.outbound.activity_manager import ActivityManager
from worker.application.ports.outbound.directory.service_directory import (
    ServiceDirectory,
)
from worker.application.ports.outbound.profiling.network.network_probe_client import (
    NetworkProbeClient,
)
from worker.application.ports.outbound.profiling.network.network_profile_publisher import (
    NetworkProfilePublisher,
)
from worker.application.profiling.contracts.network_profiling_coordinator import (
    NetworkProfilingCoordinator,
)
from worker.domain.activity.activity_request import (
    ActivityRequest,
    ActivityType,
    ResourceRequirement,
    ResourceType,
)
from worker.domain.profiling.network_profile import NetworkProfile

NETWORK_PROFILING_INTERVAL_S = 60 * 5  # 5 minutes


class DefaultNetworkProfilingCoordinator(NetworkProfilingCoordinator, AsyncLifecycle):
    class NetworkProbeClientRegistry: ...

    def __init__(
        self,
        worker_id: WorkerId,
        activity_manager: ActivityManager,
        service_resolver: ServiceDirectory,
        net_probe_client: NetworkProbeClient,
        net_profile_publisher: NetworkProfilePublisher,
    ):
        self._worker_id = worker_id
        self._activity_manager = activity_manager
        self._service_resolver = service_resolver
        self._loop_task: asyncio.Task[None] | None = None

        self._net_probe_client = net_probe_client
        self._net_profile_publisher = net_profile_publisher

        self._running_lock = asyncio.Lock()
        pass

    @override
    async def profile_network(self) -> NetworkProfile:
        ## We retrieve current active workers
        ## We acquire the grant to do network profiling
        ## We do peer-to-peer network profiling
        ## We build the network profile
        async with self._running_lock:
            worker_instances = await self._service_resolver.get_all_worker_instances()

            ## We use an independent profile for each worker
            ## In this way we avoid stopping the transmission of data
            connections = {}
            for worker_instance in worker_instances:
                activity_request = self._build_activity_request()
                activity_grant = await self._activity_manager.acquire_activity_grant(
                    activity_request
                )
                async with activity_grant:
                    connection_info = await self._net_probe_client.probe_connection(
                        worker_instance
                    )
                    connections[worker_instance.service_id] = connection_info

            return NetworkProfile(connections=connections)

    async def _coordinate_network_profiling(self) -> None:
        network_profile = await self.profile_network()
        await self._net_profile_publisher.publish_network_profile(
            self._worker_id, network_profile
        )

    @override
    async def start(self) -> None:
        ## We do periodical profiling here.
        async with asyncio.TaskGroup() as task_group:
            self._loop_task = task_group.create_task(self._profile_loop())

    async def _profile_loop(self) -> None:
        while True:
            await self._coordinate_network_profiling()
            await asyncio.sleep(NETWORK_PROFILING_INTERVAL_S)

    @override
    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()

    def _build_activity_request(self) -> ActivityRequest:

        required_lock = {
            ResourceType.NETWORK: ResourceRequirement(
                quantity=0,
                exclusive=True,
            )
        }
        activity_request = ActivityRequest(
            activity_type=ActivityType.PROFILING_NETWORK,
            required_locks=required_lock,
        )

        return activity_request
