from integration.dto.worker_profiles.resource_profile_dto import (
    CpuProfileDto,
    GpuProfileDto,
    RamProfileDto,
    ResourceProfileDto,
)
from shared.service.service import WorkerId
from worker.domain.profiling.resource_profile import ResourceProfile


class ResourceProfileMapper:
    @staticmethod
    def resource_profile_to_dto(
        worker_id: WorkerId,
        resource_profile: ResourceProfile,
    ) -> ResourceProfileDto:
        return ResourceProfileDto(
            worker_id=worker_id,
            gpu_profiles=tuple(
                GpuProfileDto(
                    device_index=profile.device_index,
                    device_uuid=profile.device_uuid,
                    name=profile.name,
                    total_memory_bytes=profile.total_memory_bytes,
                    used_memory_bytes=profile.used_memory_bytes,
                    available_memory_bytes=profile.available_memory_bytes,
                    compute_utilization_ratio=profile.compute_utilization_ratio,
                    memory_io_utilization_ratio=profile.memory_io_utilization_ratio,
                )
                for profile in resource_profile.gpu_profiles
            ),
            cpu_profile=CpuProfileDto(
                logical_cpu_count=resource_profile.cpu_profile.logical_cpu_count,
                capacity_cores=resource_profile.cpu_profile.capacity_cores,
                utilization_ratio=resource_profile.cpu_profile.utilization_ratio,
                sample_duration_s=resource_profile.cpu_profile.sample_duration_s,
            ),
            ram_profile=RamProfileDto(
                capacity_bytes=resource_profile.ram_profile.capacity_bytes,
                used_bytes=resource_profile.ram_profile.used_bytes,
                available_bytes=resource_profile.ram_profile.available_bytes,
                worker_rss_bytes=resource_profile.ram_profile.worker_rss_bytes,
            ),
        )
