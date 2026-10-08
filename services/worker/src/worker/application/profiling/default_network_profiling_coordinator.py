import asyncio
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.service.service import WorkerId
from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.directory.service_directory import (
    ServiceDirectory,
)
from worker.application.ports.outbound.profiling.network.network_probe_client import (
    NetworkProbeClient,
)
from worker.application.ports.outbound.profiling.network.network_profile_publisher import (
    NetworkProfilePublisher,
)
from worker.application.profiling.abc.network_profiling_coordinator import (
    NetworkProfilingCoordinator,
)
from worker.domain.activity.activity_request import (
    ActivityType,
)
from worker.domain.profiling.network_profile import NetworkProfile, WorkerConnection

NETWORK_PROFILING_INTERVAL_S = 60 * 5  # 5 minutes


class DefaultNetworkProfilingCoordinator(NetworkProfilingCoordinator, AsyncLifecycle):
    class NetworkProbeClientRegistry: ...

    def __init__(
        self,
        worker_id: WorkerId,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        service_directory: ServiceDirectory,
        net_probe_client: NetworkProbeClient,
        net_profile_publisher: NetworkProfilePublisher,
    ):
        self._worker_id = worker_id
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
        self._service_directory = service_directory
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
            worker_instances = await self._service_directory.get_all_worker_instances()

            activity_request = (
                self._activity_request_factory.create_request_for_activity_type(
                    ActivityType.NETWORK_PROFILING
                )
            )
            connections = []
            ## We use an independent profile for each worker
            ## In this way we avoid stopping the transmission of data
            for worker_instance in worker_instances:
                if worker_instance.service_id == self._worker_id:
                    ## We skip ourselves
                    continue

                activity_grant = await self._activity_manager.acquire_activity_grant(
                    activity_request
                )
                async with activity_grant:
                    connection_info = await self._net_probe_client.probe_connection(
                        worker_instance
                    )
                    worker_connection = WorkerConnection(
                        worker_id=worker_instance.service_id,
                        connection_info=connection_info,
                    )
                    connections.append(worker_connection)

            return NetworkProfile(connections=connections)

    async def _coordinate_network_profiling(self) -> None:
        network_profile = await self.profile_network()
        await self._net_profile_publisher.publish_network_profile(
            self._worker_id, network_profile
        )

    async def _profile_loop(self) -> None:
        while True:
            await self._coordinate_network_profiling()
            await asyncio.sleep(NETWORK_PROFILING_INTERVAL_S)

    @override
    async def start(self) -> None:
        ## We do periodical profiling here.
        self._loop_task = asyncio.create_task(self._profile_loop())

    @override
    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()
