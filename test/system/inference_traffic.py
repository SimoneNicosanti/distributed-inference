"""Generate periodic arrivals without waiting for previous responses."""

import asyncio
import math
from dataclasses import dataclass

import httpx

from integration.front_end.http.front_end_http_paths import FRONT_END_RUN_INFERENCE_PATH
from integration.front_end.http.front_end_http_schema import RunInferenceRequest


@dataclass(frozen=True)
class TrafficSample:
    request_number: int
    scheduled_at_s: float
    started_at_s: float
    response_time_s: float
    response: httpx.Response | None
    error: str | None


async def run_inference_traffic(
    client: httpx.AsyncClient,
    request: RunInferenceRequest,
    *,
    arrival_rate: float,
    request_count: int,
) -> tuple[TrafficSample, ...]:
    if not math.isfinite(arrival_rate) or arrival_rate <= 0:
        raise ValueError("arrival_rate must be finite and positive")
    if request_count < 2:
        raise ValueError("request_count must be at least 2")

    # Serialize once, before the measured arrival schedule starts.
    payload = request.model_dump_json().encode()
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def send(index: int) -> TrafficSample:
        sent = loop.time()
        response = None
        error = None
        try:
            response = await client.post(
                FRONT_END_RUN_INFERENCE_PATH,
                content=payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exception:
            error = f"{type(exception).__name__}: {exception}"
        return TrafficSample(
            request_number=index + 1,
            scheduled_at_s=index / arrival_rate,
            started_at_s=sent - started,
            response_time_s=loop.time() - sent,
            response=response,
            error=error,
        )

    tasks = []
    async with asyncio.TaskGroup() as group:
        for index in range(request_count):
            await asyncio.sleep(max(0.0, started + index / arrival_rate - loop.time()))
            tasks.append(group.create_task(send(index)))
    return tuple(task.result() for task in tasks)
