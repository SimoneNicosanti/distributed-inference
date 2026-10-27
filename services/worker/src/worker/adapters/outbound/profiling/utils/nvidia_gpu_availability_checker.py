# ruff: noqa: N802

from contextlib import suppress
from importlib import import_module
from typing import Protocol, cast, override

from worker.application.ports.outbound.profiling.utils.gpu_availability_checker import (
    GPUAvailabilityChecker,
)


class _NvmlApi(Protocol):
    NVMLError: type[BaseException]

    def nvmlInit(self) -> None: ...

    def nvmlShutdown(self) -> None: ...

    def nvmlDeviceGetCount(self) -> int: ...


class NvidiaGPUAvailabilityChecker(GPUAvailabilityChecker):
    @staticmethod
    @override
    def check() -> bool:
        try:
            nvml = cast(_NvmlApi, import_module("pynvml"))
        except ImportError:
            return False

        try:
            nvml.nvmlInit()
        except nvml.NVMLError:
            return False

        try:
            return nvml.nvmlDeviceGetCount() > 0
        except nvml.NVMLError:
            return False
        finally:
            with suppress(nvml.NVMLError):
                nvml.nvmlShutdown()
