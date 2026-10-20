from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.service.service import WorkerId


class GpuProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    device_index: int = Field(ge=0)
    device_uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)

    total_memory_bytes: int = Field(gt=0)
    used_memory_bytes: int = Field(ge=0)
    available_memory_bytes: int = Field(ge=0)

    compute_utilization_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    memory_io_utilization_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_memory(self) -> GpuProfile:
        if self.used_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU used memory exceeds total memory")
        if self.available_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU available memory exceeds total memory")
        return self


class CpuProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_cpu_count: int = Field(gt=0)
    capacity_cores: float = Field(gt=0, allow_inf_nan=False)
    utilization_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    sample_duration_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_sample(self) -> CpuProfile:
        if (self.utilization_ratio is None) != (self.sample_duration_s is None):
            raise ValueError(
                "CPU utilization and sample duration must either both be set "
                "or both be absent"
            )
        if self.capacity_cores > self.logical_cpu_count:
            raise ValueError("CPU capacity exceeds the logical CPU count")
        return self


class RamProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    capacity_bytes: int = Field(gt=0)
    used_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    worker_rss_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_memory(self) -> RamProfile:
        if self.available_bytes > self.capacity_bytes:
            raise ValueError("Available RAM exceeds RAM capacity")
        return self


class WorkerResourceProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    gpu_profiles: tuple[GpuProfile, ...]
    cpu_profile: CpuProfile
    ram_profile: RamProfile

    @model_validator(mode="after")
    def validate_gpu_identities(self) -> WorkerResourceProfile:
        device_indices = [profile.device_index for profile in self.gpu_profiles]
        if len(device_indices) != len(set(device_indices)):
            raise ValueError("Duplicate GPU device indices")

        device_uuids = [profile.device_uuid for profile in self.gpu_profiles]
        if len(device_uuids) != len(set(device_uuids)):
            raise ValueError("Duplicate GPU device UUIDs")
        return self
