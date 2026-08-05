from distributed_inference.domain.identifiers import ServerId, ServiceId


def generate_redis_directory_server_key(server_id: ServerId) -> str:
    return f"discovery:servers:{server_id.id}"


def generate_redis_directory_service_key(service_id: ServiceId) -> str:
    return f"discovery:services:{service_id.server_id.id}:{service_id.service_id}"


def generate_redis_directory_service_key_pattern_per_server(server_id: ServerId) -> str:
    return f"discovery:services:{server_id.id}:*"
