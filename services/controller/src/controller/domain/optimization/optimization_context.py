from pydantic import BaseModel, ConfigDict, model_validator

from controller.domain.profile.model_variant_execution_profile import (
    WorkerExecutionProfiles,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.network_topology import NetworkTopology
from controller.domain.profile.resource_profile import WorkerResourceProfile
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class OptimizationContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    static_profiles_by_variant: dict[ModelVariantId, ModelVariantStaticProfile]
    network_topology: NetworkTopology
    resource_profiles_by_worker: dict[WorkerId, WorkerResourceProfile]
    execution_profiles_by_worker: dict[WorkerId, WorkerExecutionProfiles]

    @model_validator(mode="after")
    def validate_indexes(self) -> OptimizationContext:
        mismatched_static_profiles = [
            model_variant_id
            for model_variant_id, profile in self.static_profiles_by_variant.items()
            if profile.model_variant_id != model_variant_id
        ]
        if mismatched_static_profiles:
            raise ValueError("Static profile keys do not match their model variant IDs")

        mismatched_resource_profiles = [
            worker_id
            for worker_id, profile in self.resource_profiles_by_worker.items()
            if profile.worker_id != worker_id
        ]
        if mismatched_resource_profiles:
            raise ValueError("Resource profile keys do not match their worker IDs")

        mismatched_execution_profiles = [
            worker_id
            for worker_id, profile in self.execution_profiles_by_worker.items()
            if profile.worker_id != worker_id
        ]
        if mismatched_execution_profiles:
            raise ValueError("Execution profile keys do not match their worker IDs")

        known_workers = self.network_topology.workers()
        indexed_workers = set(self.resource_profiles_by_worker) | set(
            self.execution_profiles_by_worker
        )
        if unknown_workers := indexed_workers - known_workers:
            unknown_worker_ids = sorted(
                str(worker_id.service_id) for worker_id in unknown_workers
            )
            raise ValueError(
                f"Profiles reference workers outside the network topology: "
                f"{unknown_worker_ids!r}"
            )

        known_model_variants = set(self.static_profiles_by_variant)
        profiled_model_variants = {
            model_variant_id
            for worker_profiles in self.execution_profiles_by_worker.values()
            for model_variant_id in worker_profiles.execution_profiles
        }
        if unknown_variants := profiled_model_variants - known_model_variants:
            unknown_variant_ids = sorted(
                model_variant_id.model_dump_json()
                for model_variant_id in unknown_variants
            )
            raise ValueError(
                "Execution profiles reference unknown model variants: "
                f"{unknown_variant_ids!r}"
            )

        return self
