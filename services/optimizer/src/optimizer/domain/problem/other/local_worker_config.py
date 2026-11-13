from dataclasses import dataclass

from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class ValidLocalWorkerConfig:
    worker_id: WorkerId
    is_inference_runner: bool


@dataclass(frozen=True, slots=True)
class NoLocalWorkerConfig:
    pass


type LocalWorkerConfig = ValidLocalWorkerConfig | NoLocalWorkerConfig
