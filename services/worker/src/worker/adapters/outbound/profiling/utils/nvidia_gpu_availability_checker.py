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
            print("Unavailable pynvml")
            return False

        try:
            nvml.nvmlInit()
        except nvml.NVMLError:
            print("No nvml Init")
            return False

        try:
            found = nvml.nvmlDeviceGetCount() > 0
            print("Found GPU")
            return found
        except nvml.NVMLError:
            print("Not Found GPU")
            return False
        finally:
            with suppress(nvml.NVMLError):
                nvml.nvmlShutdown()
