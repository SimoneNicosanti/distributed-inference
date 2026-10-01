from typing import override

from worker.application.ports.outbound.profiling.resource.gpu_profiler import (
    GpuProfiler,
)
from worker.domain.profiling.resource_profile import GpuProfile


class NoGpuProfiler(GpuProfiler):
    def __init__(self):
        pass

    @override
    async def profile_gpus(self) -> tuple[GpuProfile, ...]:
        return ()
