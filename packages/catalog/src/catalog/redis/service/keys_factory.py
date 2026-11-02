from shared.service.service import ServiceId


def build_catalog_key() -> str:
    return "catalog:services"


def build_service_hset_key(service_id: ServiceId) -> str:
    return str(service_id.id)
