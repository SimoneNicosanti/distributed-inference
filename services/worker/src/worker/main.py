import asyncio
import os
from collections.abc import Sequence
from typing import Protocol

import uvicorn
from fastapi import FastAPI
from pydantic import HttpUrl

from integration.service_instance.service_instance import (
    ArrowUrl,
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.service_instance.worker_instance import WorkerInstance
from integration.worker.http_worker_constants import DEFAULT_LOCAL_WORKER_HTTP_PORT
from shared.service.service import ServiceId
from worker.adapters.inbound.deployment.http.http_service_plan_deployer import (
    build_worker_plan_deployment_router,
)
from worker.adapters.inbound.hop_contribution.arrow.arrow_hop_invocation_contribution_server import (
    ArrowHopInvocationContributionServer,
)
from worker.adapters.inbound.profiling.arrow.arrow_network_probe_server import (
    ArrowNetworkProbeServer,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retriever import (
    ArrowModelPassResultRetriever,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_server import (
    ArrowModelPassResultServer,
)
from worker.adapters.inbound.worker_identity.http_worker_identity_query import (
    build_worker_identity_query_router,
)
from worker.composition.composition_root import WorkerComposition
from worker.composition.default_composition_root import DefaultCompositionRoot

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_ADVERTISED_HOST = "worker"
DEFAULT_HOP_INVOCATION_PORT = 8815
DEFAULT_NETWORK_PROFILING_PORT = 8816
DEFAULT_RESULT_SINK_PORT = 8817


class FlightServer(Protocol):
    def serve(self) -> None: ...

    def shutdown(self) -> None: ...


def _get_port(variable_name: str, default: int) -> int:
    port = int(os.environ.get(variable_name, default))
    if not 1 <= port <= 65535:
        raise ValueError(f"{variable_name} must be between 1 and 65535")
    return port


def _arrow_url(host: str, port: int) -> ArrowUrl:
    return ArrowUrl(f"grpc+tcp://{host}:{port}")


def _arrow_interface(host: str, port: int) -> CapabilityInterface:
    return CapabilityInterface(
        protocol=CapabilityProtocol.ARROW,
        endpoint=_arrow_url(host, port),
    )


def build_worker_instance() -> WorkerInstance:
    advertised_host = os.environ.get(
        "WORKER_ADVERTISED_HOST",
        DEFAULT_ADVERTISED_HOST,
    )
    http_port = _get_port("WORKER_HTTP_PORT", DEFAULT_LOCAL_WORKER_HTTP_PORT)
    hop_invocation_port = _get_port(
        "WORKER_HOP_INVOCATION_PORT",
        int(
            os.environ.get(
                "WORKER_PARTITION_INFERENCE_PORT",
                DEFAULT_HOP_INVOCATION_PORT,
            )
        ),
    )
    network_port = _get_port(
        "WORKER_NETWORK_PROFILING_PORT",
        DEFAULT_NETWORK_PROFILING_PORT,
    )

    capabilities = [
        Capability(
            type=CapabilityType.HOP_INVOCATION,
            interfaces=[_arrow_interface(advertised_host, hop_invocation_port)],
        ),
        Capability(
            type=CapabilityType.PLAN_ACTUATION,
            interfaces=[
                CapabilityInterface(
                    protocol=CapabilityProtocol.HTTP,
                    endpoint=HttpUrl(f"http://{advertised_host}:{http_port}"),
                )
            ],
        ),
        Capability(
            type=CapabilityType.NETWORK_PROFILING,
            interfaces=[_arrow_interface(advertised_host, network_port)],
        ),
        Capability(
            type=CapabilityType.EXECUTION_PROFILING,
            interfaces=[],
        ),
        Capability(
            type=CapabilityType.RESOURCE_PROFILING,
            interfaces=[],
        ),
        Capability(
            type=CapabilityType.RESULT_SINKER,
            interfaces=[
                _arrow_interface(
                    advertised_host,
                    _get_port("WORKER_RESULT_SINK_PORT", DEFAULT_RESULT_SINK_PORT),
                )
            ],
        ),
    ]

    return WorkerInstance(
        service_id=ServiceId(),
        capabilities=capabilities,
    )


def build_http_app(composition: WorkerComposition) -> FastAPI:
    app = FastAPI(title="Worker")
    app.include_router(
        build_worker_plan_deployment_router(composition.worker_plan_deployer)
    )
    app.include_router(
        build_worker_identity_query_router(composition.worker_identity_query)
    )
    return app


def build_flight_servers(
    composition: WorkerComposition,
    event_loop: asyncio.AbstractEventLoop,
) -> list[FlightServer]:
    bind_host = os.environ.get("WORKER_BIND_HOST", DEFAULT_BIND_HOST)
    servers: list[FlightServer] = [
        ArrowHopInvocationContributionServer(
            _arrow_url(
                bind_host,
                _get_port(
                    "WORKER_HOP_INVOCATION_PORT",
                    int(
                        os.environ.get(
                            "WORKER_PARTITION_INFERENCE_PORT",
                            DEFAULT_HOP_INVOCATION_PORT,
                        )
                    ),
                ),
            ).unicode_string(),
            composition.hop_contribution_receiver,
            event_loop,
        ),
        ArrowNetworkProbeServer(
            _arrow_url(
                bind_host,
                _get_port(
                    "WORKER_NETWORK_PROFILING_PORT",
                    DEFAULT_NETWORK_PROFILING_PORT,
                ),
            ).unicode_string()
        ),
    ]

    servers.append(
        ArrowModelPassResultServer(
            _arrow_url(
                bind_host,
                _get_port("WORKER_RESULT_SINK_PORT", DEFAULT_RESULT_SINK_PORT),
            ).unicode_string(),
            ArrowModelPassResultRetriever(
                composition.model_pass_result_retriever,
                event_loop,
            ),
        )
    )

    return servers


async def serve(
    http_server: uvicorn.Server,
    flight_servers: Sequence[FlightServer],
) -> None:
    tasks = [
        asyncio.create_task(http_server.serve(), name="worker-http-server"),
        *[
            asyncio.create_task(
                asyncio.to_thread(server.serve),
                name=f"worker-flight-server-{index}",
            )
            for index, server in enumerate(flight_servers)
        ],
    ]

    try:
        completed, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in completed:
            task.result()
    finally:
        http_server.should_exit = True
        await asyncio.gather(
            *(asyncio.to_thread(server.shutdown) for server in flight_servers),
            return_exceptions=True,
        )
        await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    worker_instance = build_worker_instance()
    composition_root = DefaultCompositionRoot()

    async with composition_root.compose(worker_instance) as composition:
        http_port = _get_port("WORKER_HTTP_PORT", DEFAULT_LOCAL_WORKER_HTTP_PORT)
        bind_host = os.environ.get("WORKER_BIND_HOST", DEFAULT_BIND_HOST)
        http_server = uvicorn.Server(
            uvicorn.Config(
                build_http_app(composition),
                host=bind_host,
                port=http_port,
            )
        )
        flight_servers = build_flight_servers(
            composition,
            asyncio.get_running_loop(),
        )
        await serve(http_server, flight_servers)


if __name__ == "__main__":
    asyncio.run(main())
