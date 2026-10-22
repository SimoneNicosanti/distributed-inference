import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuProfile:
    device_index: int
    device_uuid: str
    name: str

    total_memory_bytes: int
    used_memory_bytes: int
    available_memory_bytes: int

    compute_utilization_ratio: float | None
    memory_io_utilization_ratio: float | None

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("GPU device index must be non-negative")
        if not self.device_uuid or not self.name:
            raise ValueError("GPU identity fields must not be empty")
        if self.total_memory_bytes <= 0:
            raise ValueError("GPU total memory must be positive")
        if self.used_memory_bytes < 0 or self.available_memory_bytes < 0:
            raise ValueError("GPU memory values must be non-negative")
        if self.used_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU used memory exceeds total memory")
        if self.available_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU available memory exceeds total memory")
        for ratio in (
            self.compute_utilization_ratio,
            self.memory_io_utilization_ratio,
        ):
            if ratio is not None and (not math.isfinite(ratio) or not 0 <= ratio <= 1):
                raise ValueError("GPU utilization ratios must be between zero and one")


@dataclass(frozen=True, slots=True)
class CpuProfile:
    logical_cpu_count: int
    capacity_cores: float
    utilization_ratio: float | None
    sample_duration_s: float | None

    def __post_init__(self) -> None:
        if self.logical_cpu_count <= 0:
            raise ValueError("Logical CPU count must be positive")
        if not math.isfinite(self.capacity_cores) or self.capacity_cores <= 0:
            raise ValueError("CPU capacity must be finite and positive")
        if (self.utilization_ratio is None) != (self.sample_duration_s is None):
            raise ValueError(
                "CPU utilization and sample duration must either both be set "
                "or both be absent"
            )
        if self.capacity_cores > self.logical_cpu_count:
            raise ValueError("CPU capacity exceeds the logical CPU count")
        if self.utilization_ratio is not None and (
            not math.isfinite(self.utilization_ratio)
            or not 0 <= self.utilization_ratio <= 1
        ):
            raise ValueError("CPU utilization ratio must be between zero and one")
        if self.sample_duration_s is not None and (
            not math.isfinite(self.sample_duration_s) or self.sample_duration_s <= 0
        ):
            raise ValueError("CPU sample duration must be finite and positive")


@dataclass(frozen=True, slots=True)
class RamProfile:
    capacity_bytes: int
    used_bytes: int
    available_bytes: int
    worker_rss_bytes: int

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError("RAM capacity must be positive")
        if any(
            value < 0
            for value in (self.used_bytes, self.available_bytes, self.worker_rss_bytes)
        ):
            raise ValueError("RAM usage values must be non-negative")
        if self.available_bytes > self.capacity_bytes:
            raise ValueError("Available RAM exceeds RAM capacity")


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    gpu_profiles: tuple[GpuProfile, ...]
    cpu_profile: CpuProfile
    ram_profile: RamProfile
