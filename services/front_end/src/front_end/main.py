import asyncio
import os

import uvicorn
from fastapi import FastAPI

from front_end.adapters.inbound.http_front_end_router import build_front_end_router
from front_end.composition.composition_root import FrontEndComposition
from front_end.composition.default_composition_root import DefaultCompositionRoot

DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8000


def build_http_app(composition: FrontEndComposition) -> FastAPI:
    app = FastAPI(title="Front End")
    app.include_router(
        build_front_end_router(
            inference_runner=composition.inference_runner,
            local_worker_identity_query=composition.local_worker_identity_query,
            model_loader=composition.model_loader,
        )
    )
    return app


async def main() -> None:
    composition_root = DefaultCompositionRoot()

    async with composition_root.compose() as composition:
        server = uvicorn.Server(
            uvicorn.Config(
                build_http_app(composition),
                host=os.environ.get("FRONT_END_HTTP_HOST", DEFAULT_HTTP_HOST),
                port=int(os.environ.get("FRONT_END_HTTP_PORT", DEFAULT_HTTP_PORT)),
            )
        )
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
