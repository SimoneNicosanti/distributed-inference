from typing import Self

from pydantic import ConfigDict, model_validator

from directory.contracts.service_instance import CapabilityType, ServiceInstance


class WorkerInstance(ServiceInstance):
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:

        if len(self.capabilities) == 0:
            raise ValueError("Worker instance must have at least one capability")

        required_capabilities = {
            CapabilityType.INFERENCE_SERVICE,
            CapabilityType.NETWORK_PROFILING,
            CapabilityType.EXECUTION_PROFILING,
        }

        actual_capabilities = {capability.type for capability in self.capabilities}

        missing_capabilities = required_capabilities - actual_capabilities
        if len(missing_capabilities) > 0:
            raise ValueError(
                f"Worker instance is missing the following capabilities: {missing_capabilities}"
            )

        return self
