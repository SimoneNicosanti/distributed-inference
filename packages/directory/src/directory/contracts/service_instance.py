from enum import StrEnum, auto
from typing import Self

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    HttpUrl,
    TypeAdapter,
    model_validator,
)

from shared.service.service import ServiceId


class CapabilityProtocol(StrEnum):
    GRPC = auto()
    HTTP = auto()
    PYRO = auto()
    ARROW = auto()
    ...


type OperationName = str  ## String representing the endpoint method name
type EndpointUrl = str  ## String representing the endpoint


## TODO: Do actual validation
class PyroUrl(AnyUrl):
    @classmethod
    def validate(cls, v: AnyUrl) -> AnyUrl:
        return v


## TODO: Do actual validation
class ArrowUrl(AnyUrl):
    @classmethod
    def validate(cls, v: AnyUrl) -> AnyUrl:
        return v


## TODO: Do actual validation
class GrpcUrl(AnyUrl):
    @classmethod
    def validate(cls, v: AnyUrl) -> AnyUrl:
        return v


class CapabilityInterface(BaseModel):
    ## TODO: Add validation for host and port
    model_config = ConfigDict(frozen=True)

    protocol: CapabilityProtocol
    endpoint: HttpUrl | PyroUrl | ArrowUrl | GrpcUrl

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        match self.protocol:
            case CapabilityProtocol.HTTP:
                validator = TypeAdapter(HttpUrl)
            case CapabilityProtocol.PYRO:
                validator = TypeAdapter(PyroUrl)
            case CapabilityProtocol.ARROW:
                validator = TypeAdapter(ArrowUrl)
            case CapabilityProtocol.GRPC:
                validator = TypeAdapter(GrpcUrl)
            case _:
                raise ValueError(f"Invalid protocol {self.protocol}")

        validator.validate_python(self.endpoint)

        return self


class CapabilityType(StrEnum):
    ## Declared by model manager
    MODEL_PARTITION = auto()
    MODEL_REGISTRATION = auto()

    ## Declared by deployment optimization
    DEPLOYMENT_OPTIMIZATION = auto()

    ## Declared by artifact storage (e.g. S3)
    ARTIFACT_STORAGE = auto()

    ## Profile storage
    PROFILE_STORAGE = auto()

    ## Declared by worker
    PARTITION_INFERENCE = auto()
    DEPLOYMENT_ACTUATION = auto()
    NETWORK_PROFILING = auto()
    EXECUTION_PROFILING = auto()
    RESOURCE_PROFILING = auto()

    # Declared by ingress
    INVOCATION_OUTPUT = auto()


class Capability(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: CapabilityType
    interfaces: list[CapabilityInterface]

    def get_interfaces_by_protocol(
        self, protocol: CapabilityProtocol
    ) -> list[CapabilityInterface]:
        return list(filter(lambda x: x.protocol == protocol, self.interfaces))


class ServiceInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: ServiceId
    capabilities: list[Capability]

    def get_capabilities_by_type(
        self, capability_type: CapabilityType
    ) -> list[Capability]:
        return list(filter(lambda x: x.type == capability_type, self.capabilities))

    def get_capability_interfaces_by_type_and_protocol(
        self, capability_type: CapabilityType, protocol: CapabilityProtocol
    ) -> list[CapabilityInterface]:

        capabilities = self.get_capabilities_by_type(capability_type)
        interfaces = []
        for capability in capabilities:
            interfaces.extend(capability.get_interfaces_by_protocol(protocol))

        return interfaces
