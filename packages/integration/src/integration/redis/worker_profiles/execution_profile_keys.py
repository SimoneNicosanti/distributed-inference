from shared.service.service import WorkerId


def build_execution_profile_key_per_worker(worker_id: WorkerId) -> str:
    return f"worker:{worker_id.service_id}:model-execution-profiles"


def build_execution_profile_ids_key_per_worker(worker_id: WorkerId) -> str:
    return f"worker:{worker_id.service_id}:model-execution-profile-ids"


def build_whole_execution_profile_key() -> str:
    return "worker:*:model-execution-profiles"
