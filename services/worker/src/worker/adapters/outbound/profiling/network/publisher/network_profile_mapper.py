from integration.dto.worker_profiles.network_profile_dto import (
    NetworkConnectionDto,
    NetworkProfileDto,
)
from shared.service.service import WorkerId
from worker.domain.profiling.network_profile import NetworkProfile


class NetworkProfileMapper:
    @staticmethod
    def network_profile_to_dto(
        worker_id: WorkerId,
        network_profile: NetworkProfile,
    ):
        return NetworkProfileDto(
            source_worker_id=worker_id,
            connections=tuple(
                NetworkConnectionDto(
                    target_worker_id=connection.worker_id,
                    throughput_bytes_per_sec=(
                        connection.connection_info.throughput_bytes_per_sec
                    ),
                    round_trip_time_s=connection.connection_info.round_trip_time_s,
                )
                for connection in network_profile.connections
            ),
        )
