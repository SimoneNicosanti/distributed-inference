from shared.service.service import WorkerId


def build_resource_profile_key_per_worker(worker_id: WorkerId) -> str:
    return f"worker:{worker_id.service_id}:resource-profile"


def build_whole_resource_profile_key() -> str:
    return "worker:*:resource-profile"
