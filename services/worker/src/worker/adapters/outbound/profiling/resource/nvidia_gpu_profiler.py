# ruff: noqa: N802

from importlib import import_module
from types import TracebackType
from typing import Protocol, cast, override

from worker.application.ports.outbound.profiling.resource.gpu_profiler import (
    GpuProfiler,
)
from worker.domain.profiling.resource_profile import GpuProfile


class _MemoryInfo(Protocol):
    total: int
    used: int
    free: int


class _Utilization(Protocol):
    gpu: int | None
    memory: int | None


class _NvmlApi(Protocol):
    NVMLError_NotSupported: type[BaseException]

    def nvmlInit(self) -> None: ...

    def nvmlShutdown(self) -> None: ...

    def nvmlDeviceGetCount(self) -> int: ...

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...

    def nvmlDeviceGetUUID(self, handle: object) -> str | bytes: ...

    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _MemoryInfo: ...

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> _Utilization: ...


class NvidiaGpuProfiler(GpuProfiler):
    def __init__(self, nvml_api: _NvmlApi | None = None) -> None:
        self._nvml = (
            cast(_NvmlApi, import_module("pynvml"))
            if nvml_api is None
            else nvml_api
        )
        self._initialized = False
        self._nvml.nvmlInit()
        self._initialized = True

    @override
    async def profile_gpus(self) -> tuple[GpuProfile, ...]:
        if not self._initialized:
            raise RuntimeError("The NVIDIA GPU profiler is closed")

        device_count = self._nvml.nvmlDeviceGetCount()
        if device_count < 0:
            raise ValueError("NVML returned a negative GPU count")

        return tuple(
            self._profile_gpu(device_index)
            for device_index in range(device_count)
        )

    def close(self) -> None:
        if not self._initialized:
            return

        self._nvml.nvmlShutdown()
        self._initialized = False

    def __enter__(self) -> NvidiaGpuProfiler:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _profile_gpu(self, device_index: int) -> GpuProfile:
        handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
        memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        self._validate_memory(memory, device_index)

        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            compute_utilization_ratio = self._percentage_to_ratio(utilization.gpu)
            memory_io_utilization_ratio = self._percentage_to_ratio(
                utilization.memory
            )
        except self._nvml.NVMLError_NotSupported:
            compute_utilization_ratio = None
            memory_io_utilization_ratio = None

        return GpuProfile(
            device_index=device_index,
            device_uuid=self._decode_text(self._nvml.nvmlDeviceGetUUID(handle)),
            name=self._decode_text(self._nvml.nvmlDeviceGetName(handle)),
            total_memory_bytes=memory.total,
            used_memory_bytes=memory.used,
            available_memory_bytes=memory.free,
            compute_utilization_ratio=compute_utilization_ratio,
            memory_io_utilization_ratio=memory_io_utilization_ratio,
        )

    @staticmethod
    def _validate_memory(memory: _MemoryInfo, device_index: int) -> None:
        if (
            memory.total <= 0
            or memory.used < 0
            or memory.free < 0
            or memory.used > memory.total
            or memory.free > memory.total
        ):
            raise ValueError(f"NVML returned invalid memory for GPU {device_index}")

    @staticmethod
    def _percentage_to_ratio(percentage: int | None) -> float | None:
        if percentage is None:
            return None
        if not 0 <= percentage <= 100:
            raise ValueError(f"NVML returned an invalid utilization: {percentage}")
        return percentage / 100

    @staticmethod
    def _decode_text(value: str | bytes) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value
