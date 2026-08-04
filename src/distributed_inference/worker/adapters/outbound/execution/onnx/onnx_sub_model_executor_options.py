from dataclasses import dataclass
from enum import StrEnum, auto


class OnnxDeviceType(StrEnum):
    CPU = auto()
    CUDA = auto()


@dataclass(frozen=True)
class OnnxSubModelExecutorOptions:
    device_type: OnnxDeviceType
    device_id: int

    def __post_init__(self) -> None:
        if self.device_type == OnnxDeviceType.CUDA and self.device_id < 0:
            raise ValueError("CUDA device id must be >= 0")

        elif self.device_type == OnnxDeviceType.CPU and self.device_id != 0:
            raise ValueError("CPU device id must be 0")
