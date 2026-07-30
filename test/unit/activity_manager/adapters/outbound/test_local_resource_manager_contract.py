from typing import override

from distributed_inference.activity_manager.adapters.outbound.local_resource_manager import (
    LocalResourceManager,
)
from distributed_inference.activity_manager.application.ports.outbound.resource_manager import (
    ResourceManager,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceAvailability,
)
from test.contracts.activity_manager.resource_manager_contract import (
    ResourceManagerContract,
)


class TestLocalResourceManagerContract(ResourceManagerContract):
    @override
    def build_resource_manager(
        self,
        resource_availability: ResourceAvailability,
    ) -> ResourceManager:
        return LocalResourceManager(resource_availability)
