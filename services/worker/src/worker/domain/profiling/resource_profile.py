from pydantic import BaseModel, ConfigDict


class GpuProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    device_index: int
    device_uuid: str
    name: str

    total_memory_bytes: int
    used_memory_bytes: int
    available_memory_bytes: int

    compute_utilization_ratio: float | None
    memory_io_utilization_ratio: float | None


class CpuProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_cpu_count: int
    capacity_cores: float
    utilization_ratio: float | None
    sample_duration_s: float | None


class RamProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    capacity_bytes: int
    used_bytes: int
    available_bytes: int
    worker_rss_bytes: int


class ResourceProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_profiles: tuple[GpuProfile, ...]
    cpu_profile: CpuProfile
    ram_profile: RamProfile
