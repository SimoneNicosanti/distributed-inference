from abc import ABC
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

from integration.identifiers.identifiers import ServiceId

## TODO: Should fix this to declare just the whole endpoint and not everything f


class ServiceProtocol(StrEnum):
    GRPC = auto()
    HTTP = auto()
    PYRO = auto()
    ARROW = auto()
    ...


type EndpointValue = str  ## String representing the endpoint
type EndpointKey = str  ## String representing the endpoint method name


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


class ServiceEndpoint(BaseModel, ABC):
    ## TODO: Add validation for host and port
    model_config = ConfigDict(frozen=True)

    protocol: ServiceProtocol
    endpoints: dict[EndpointKey, EndpointValue]

    @model_validator(mode="after")
    def validate_endpoints(self) -> Self:
        for key, value in self.endpoints.items():
            match self.protocol:
                case ServiceProtocol.HTTP:
                    validator = TypeAdapter(HttpUrl)
                case ServiceProtocol.PYRO:
                    validator = TypeAdapter(PyroUrl)
                case ServiceProtocol.ARROW:
                    validator = TypeAdapter(ArrowUrl)
                case ServiceProtocol.GRPC:
                    validator = TypeAdapter(GrpcUrl)
                case _:
                    raise ValueError(f"Invalid protocol {self.protocol}")

            validator.validate_python(value)

        return self

    def get_endpoint_by_key(self, key: EndpointKey) -> EndpointValue:
        if key not in self.endpoints:
            raise ValueError(f"Endpoint {key} not found")
        return self.endpoints[key]


class ServiceType(StrEnum):
    MODEL_MANAGER = auto()
    INFERENCE_SERVICE = auto()
    CONTROLLER = auto()
    ARTIFACT_STORE = auto()


class ServiceInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: ServiceId
    service_type: ServiceType

    service_endpoints: list[ServiceEndpoint]
