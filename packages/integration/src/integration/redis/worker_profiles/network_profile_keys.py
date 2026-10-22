from shared.service.service import WorkerId


def build_network_profile_key_per_worker(worker_id: WorkerId) -> str:
    return f"worker:{worker_id.service_id}:network-profile"


def build_whole_network_profile_key() -> str:
    return "worker:*:network-profile"
