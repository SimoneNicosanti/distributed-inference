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

from shared.identifiers.identifiers import ServiceId


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
    MODEL_MANAGER = auto()
    CONTROLLER = auto()
    ARTIFACT_STORE = auto()

    INFERENCE_SERVICE = auto()
    NETWORK_PROFILING = auto()
    EXECUTION_PROFILING = auto()


class ServiceCapabilities(BaseModel):
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
    capabilities: list[ServiceCapabilities]

    def get_capability_by_type(
        self, capability_type: CapabilityType
    ) -> list[ServiceCapabilities]:
        return list(filter(lambda x: x.type == capability_type, self.capabilities))
