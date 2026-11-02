"""Stable homogeneous worker pools and the explicit VRAM-to-RAM projection."""

from dataclasses import dataclass

from controller.domain.profile.resource_profile import (
    CpuProfile,
    GpuProfile,
    RamProfile,
    WorkerResourceProfile,
)
from scalability_benchmark.configuration import ExperimentConfig, TierSettings
from scalability_benchmark.randomness import stable_uuid
from shared.service.service import ServiceId, WorkerId


@dataclass(frozen=True)
class SyntheticWorker:
    tier: str
    index: int
    hardware: TierSettings
    resource: WorkerResourceProfile

    @property
    def id(self) -> WorkerId:
        return self.resource.worker_id

    def as_record(self) -> dict[str, object]:
        return {
            "id": str(self.id.id),
            "tier": self.tier,
            "index": self.index,
            "physical_hardware": self.hardware.model_dump(mode="json"),
            "solver_memory_capacity_bytes": self.resource.ram_profile.available_bytes,
            "solver_backend_slot": "cpu",
            "simulated_backend": self.hardware.backend,
        }


def worker_pool(config: ExperimentConfig, seed: int) -> tuple[SyntheticWorker, ...]:
    workers = []
    for tier, count in (("device", 1), ("edge", 5), ("cloud", 5)):
        hardware = config.hardware[tier]
        memory = int(
            (hardware.vram_gib if hardware.backend == "gpu" else hardware.ram_gib)
            * 2**30
        )
        for index in range(count):
            identity = stable_uuid(seed, "worker", f"{tier}-{index}")
            gpu = (
                ()
                if hardware.backend == "cpu"
                else (
                    GpuProfile(
                        0, str(identity), f"synthetic-{tier}", memory, 0, memory
                    ),
                )
            )
            workers.append(
                SyntheticWorker(
                    tier,
                    index,
                    hardware,
                    WorkerResourceProfile(
                        ServiceId(identity),
                        gpu,
                        CpuProfile(hardware.cpu_cores, float(hardware.cpu_cores)),
                        RamProfile(memory, 0, memory, 0),
                    ),
                )
            )
    return tuple(workers)


def select_workers(
    pool: tuple[SyntheticWorker, ...], edge: int, cloud: int
) -> tuple[SyntheticWorker, ...]:
    return tuple(
        worker
        for worker in pool
        if worker.tier == "device"
        or worker.index < {"edge": edge, "cloud": cloud}.get(worker.tier, 0)
    )
