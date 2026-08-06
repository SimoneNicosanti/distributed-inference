from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Annotated, Literal, override

from pydantic import BaseModel, ConfigDict, Field, field_validator

from distributed_inference.domain.identifiers import ServiceId


class ServiceProtocol(StrEnum):
    GRPC = auto()
    HTTP = auto()
    PYRO = auto()
    ARROW = auto()
    ...


class ServiceEndpoint(BaseModel, ABC):
    ## TODO: Add validation for host and port
    model_config = ConfigDict(frozen=True)

    host: str
    port: Annotated[int, Field(ge=1024, le=65535)]
    protocol: ServiceProtocol

    @field_validator("host")
    @classmethod
    def validate_host(cls, host: str) -> str:

        def is_valid_hostname(hostname: str) -> bool:
            import re

            if not hostname or len(hostname) > 253:
                return False

            label_pattern = re.compile(
                r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                re.IGNORECASE,
            )

            return all(label_pattern.fullmatch(label) for label in hostname.split("."))

        def is_valid_ip_addr(ipaddr: str) -> bool:
            import ipaddress

            try:
                ipaddress.ip_address(ipaddr)
                return True
            except ValueError:
                return False

        if not is_valid_hostname(host) and not is_valid_ip_addr(host):
            raise ValueError(f"Invalid host {host}")

        return host

    @abstractmethod
    def get_endpoint_string(self) -> str: ...


class HostServiceEndpoint(ServiceEndpoint):
    model_config = ConfigDict(frozen=True)
    kind: Literal["host_port"] = "host_port"

    protocol: Literal[
        ServiceProtocol.HTTP,
        ServiceProtocol.GRPC,
        ServiceProtocol.ARROW,
    ]

    @override
    def get_endpoint_string(self) -> str:
        match self.protocol:
            case ServiceProtocol.HTTP:
                return f"http://{self.host}:{self.port}"
            case ServiceProtocol.GRPC:
                return f"{self.host}:{self.port}"
            case ServiceProtocol.ARROW:
                return f"grpc://{self.host}:{self.port}"


class UriServiceEndpoint(ServiceEndpoint):
    model_config = ConfigDict(frozen=True)
    kind: Literal["uri"] = "uri"

    protocol: Literal[ServiceProtocol.PYRO]
    object_identifier: Annotated[str, Field(min_length=1)]

    @override
    def get_endpoint_string(self) -> str:
        match self.protocol:
            case ServiceProtocol.PYRO:
                return f"PYRO:{self.object_identifier}@{self.host}:{self.port}"


ServiceEndpointType = Annotated[
    HostServiceEndpoint | UriServiceEndpoint,
    Field(discriminator="kind"),
]


class ServiceType(StrEnum):
    MODEL_MANAGER = auto()
    INFERENCE_SERVICE = auto()
    ARTIFACT_STORE = auto()


class ServiceInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: ServiceId
    service_type: ServiceType

    service_endpoint: ServiceEndpointType
