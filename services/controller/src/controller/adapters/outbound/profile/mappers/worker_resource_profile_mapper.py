from controller.domain.profile.resource_profile import (
    CpuProfile,
    GpuProfile,
    RamProfile,
    WorkerResourceProfile,
)
from integration.dto.worker_profiles.resource_profile_dto import ResourceProfileDto


class WorkerResourceProfileMapper:
    @staticmethod
    def worker_resource_profile_from_dto(
        dto: ResourceProfileDto,
    ) -> WorkerResourceProfile:
        return WorkerResourceProfile(
            worker_id=dto.worker_id,
            gpu_profiles=tuple(
                GpuProfile(
                    device_index=profile.device_index,
                    device_uuid=profile.device_uuid,
                    name=profile.name,
                    total_memory_bytes=profile.total_memory_bytes,
                    used_memory_bytes=profile.used_memory_bytes,
                    available_memory_bytes=profile.available_memory_bytes,
                    compute_utilization_ratio=(profile.compute_utilization_ratio),
                    memory_io_utilization_ratio=(profile.memory_io_utilization_ratio),
                )
                for profile in dto.gpu_profiles
            ),
            cpu_profile=CpuProfile(
                logical_cpu_count=dto.cpu_profile.logical_cpu_count,
                capacity_cores=dto.cpu_profile.capacity_cores,
                utilization_ratio=dto.cpu_profile.utilization_ratio,
                sample_duration_s=dto.cpu_profile.sample_duration_s,
            ),
            ram_profile=RamProfile(
                capacity_bytes=dto.ram_profile.capacity_bytes,
                used_bytes=dto.ram_profile.used_bytes,
                available_bytes=dto.ram_profile.available_bytes,
                worker_rss_bytes=dto.ram_profile.worker_rss_bytes,
            ),
        )
