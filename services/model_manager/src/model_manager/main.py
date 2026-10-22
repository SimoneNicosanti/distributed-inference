import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import HttpUrl

from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from model_manager.adapters.inbound.http.http_model_manager import (
    build_model_manager_router,
)
from model_manager.composition.default_composition_root import DefaultCompositionRoot
from shared.service.service import ServiceId

# Inizializza qui le implementazioni concrete


def build_service_instance() -> ServiceInstance:

    model_partition_capability = Capability(
        type=CapabilityType.MODEL_PARTITION,
        interfaces=[
            CapabilityInterface(
                protocol=CapabilityProtocol.HTTP,
                endpoint=HttpUrl("http://model-manager:8000"),
            ),
        ],
    )

    model_registration_capability = Capability(
        type=CapabilityType.MODEL_REGISTRATION,
        interfaces=[
            CapabilityInterface(
                protocol=CapabilityProtocol.HTTP,
                endpoint=HttpUrl("http://model-manager:8000"),
            ),
        ],
    )

    return ServiceInstance(
        service_id=ServiceId(),
        capabilities=[
            model_partition_capability,
            model_registration_capability,
        ],
    )


async def main() -> None:
    composition_root = DefaultCompositionRoot()

    service_instance = build_service_instance()

    async with composition_root.compose(service_instance) as (
        model_registry,
        model_variant_registry,
        model_partitioner,
    ):
        app = FastAPI(title="Model Manager")

        app.include_router(
            build_model_manager_router(
                model_registry=model_registry,
                model_variant_registry=model_variant_registry,
                model_partitioner=model_partitioner,
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
