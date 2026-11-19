import asyncio
import os

import uvicorn
from fastapi import FastAPI, Response
from pydantic import HttpUrl

from advertisement.http.http_capability_life_checker import (
    DEFAULT_HTTP_READINESS_PATH,
)
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.composition.default_composition_root import DefaultCompositionRoot
from shared.service.service import ServiceId

DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8000
DEFAULT_ADVERTISED_HOST = "optimizer"


async def readiness() -> Response:
    return Response(status_code=204)


def build_service_instance() -> ServiceInstance:
    host = os.environ.get("OPTIMIZER_ADVERTISED_HOST", DEFAULT_ADVERTISED_HOST)
    port = int(os.environ.get("OPTIMIZER_HTTP_PORT", DEFAULT_HTTP_PORT))
    return ServiceInstance(
        service_id=ServiceId(),
        capabilities=[
            Capability(
                type=CapabilityType.OPTIMIZATION,
                interfaces=[
                    CapabilityInterface(
                        protocol=CapabilityProtocol.HTTP,
                        endpoint=HttpUrl(f"http://{host}:{port}"),
                    ),
                ],
            ),
        ],
    )


def build_http_app(optimizer_controller: OptimizerController) -> FastAPI:
    app = FastAPI(title="Optimizer")
    app.add_api_route(
        DEFAULT_HTTP_READINESS_PATH,
        readiness,
        methods=["GET"],
        status_code=204,
    )
    app.include_router(build_optimizer_router(optimizer_controller))
    return app


async def main() -> None:
    composition_root = DefaultCompositionRoot()
    service_instance = build_service_instance()
    async with composition_root.compose(service_instance) as optimizer_controller:
        config = uvicorn.Config(
            build_http_app(optimizer_controller),
            host=os.environ.get("OPTIMIZER_HTTP_HOST", DEFAULT_HTTP_HOST),
            port=int(os.environ.get("OPTIMIZER_HTTP_PORT", DEFAULT_HTTP_PORT)),
        )
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
