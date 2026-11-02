from catalog.api.service.service_directory import ServiceDirectory
from integration.service_instance.service_instance import (
    CapabilityProtocol,
    CapabilityType,
)
from shared.service.service import WorkerId


async def get_arrow_worker_endpoint(
    directory: ServiceDirectory,
    worker_id: WorkerId,
    capability: CapabilityType,
) -> str:
    instance = await directory.get_service_instance_by_service_id(worker_id)
    if instance.service_id != worker_id:
        raise ValueError("Discovered worker identity does not match requested worker")
    interfaces = instance.get_capability_interfaces_by_type_and_protocol(
        capability, CapabilityProtocol.ARROW
    )
    if len(interfaces) != 1:
        raise ValueError(
            f"Worker {worker_id} must expose exactly one Arrow {capability} interface"
        )
    return interfaces[0].endpoint.unicode_string()
