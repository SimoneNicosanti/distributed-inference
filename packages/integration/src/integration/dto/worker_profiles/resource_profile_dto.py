from pydantic import BaseModel, ConfigDict

from shared.service.service import WorkerId


class GpuProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    device_index: int
    device_uuid: str
    name: str
    total_memory_bytes: int
    used_memory_bytes: int
    available_memory_bytes: int
    compute_utilization_ratio: float | None
    memory_io_utilization_ratio: float | None


class CpuProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_cpu_count: int
    capacity_cores: float
    utilization_ratio: float | None
    sample_duration_s: float | None


class RamProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    capacity_bytes: int
    used_bytes: int
    available_bytes: int
    worker_rss_bytes: int


class ResourceProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    gpu_profiles: tuple[GpuProfileDto, ...]
    cpu_profile: CpuProfileDto
    ram_profile: RamProfileDto
