import asyncio

import uvicorn
from fastapi import FastAPI

from model_manager.adapters.inbound.http.model_manager_http_router import (
    build_model_manager_router,
)
from model_manager.composition.default_composition_root import DefaultCompositionRoot

# Inizializza qui le implementazioni concrete


async def main() -> None:
    composition_root = DefaultCompositionRoot()

    async with composition_root.compose() as (
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
