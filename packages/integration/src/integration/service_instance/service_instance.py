from enum import StrEnum, auto
from typing import Self

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    HttpUrl,
    TypeAdapter,
    UrlConstraints,
    model_validator,
)

from shared.service.service import ServiceId


class CapabilityProtocol(StrEnum):
    HTTP = auto()
    ARROW = auto()
    ...


class ArrowUrl(AnyUrl):
    _constraints = UrlConstraints(
        allowed_schemes=["grpc+tcp"],
        host_required=True,
    )


class CapabilityInterface(BaseModel):
    model_config = ConfigDict(frozen=True)

    protocol: CapabilityProtocol
    endpoint: HttpUrl | ArrowUrl

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        match self.protocol:
            case CapabilityProtocol.HTTP:
                validator = TypeAdapter(HttpUrl)
            case CapabilityProtocol.ARROW:
                validator = TypeAdapter(ArrowUrl)
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
    HOP_INVOCATION = auto()
    PLAN_ACTUATION = auto()
    IN_NETWORK_PROFILING = auto()
    RESULT_RETRIEVE = auto()


class CapabilityCouple(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: CapabilityType
    interface: CapabilityInterface


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
