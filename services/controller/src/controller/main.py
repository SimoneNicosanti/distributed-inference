import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import HttpUrl

from controller.adapters.inbound.http_optimization_manager import (
    build_optimization_manager_router,
)
from controller.composition.default_composition_root import DefaultCompositionRoot
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from shared.service.service import ServiceId


def build_service_instance() -> ServiceInstance:

    optimization_capability = Capability(
        type=CapabilityType.DEPLOYMENT_OPTIMIZATION,
        interfaces=[
            CapabilityInterface(
                protocol=CapabilityProtocol.HTTP,
                endpoint=HttpUrl("http://controller:8000"),
            ),
        ],
    )

    return ServiceInstance(
        service_id=ServiceId(),
        capabilities=[
            optimization_capability,
        ],
    )


async def main() -> None:
    composition_root = DefaultCompositionRoot()

    service_instance = build_service_instance()

    async with composition_root.compose(service_instance) as optimization_manager:
        app = FastAPI(title="Model Manager")

        app.include_router(
            build_optimization_manager_router(
                optimization_manager=optimization_manager,
            )
        )

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
        )

        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
