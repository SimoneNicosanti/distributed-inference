from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, field_validator

from distributed_inference.domain.identifiers import ServiceId


class ServiceProtocol(StrEnum):
    GRPC = auto()
    HTTP = auto()
    ...


class ServiceEndpoint(BaseModel):
    ## TODO: Add validation for host and port
    model_config = ConfigDict(frozen=True)

    host: str
    port: int
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
            raise ValueError("Invalid host")

        return host

    @field_validator("port")
    @classmethod
    def validate_port(cls, port: int) -> int:

        ## Allowing only not reserved ports
        if not 1024 <= port <= 65535:
            raise ValueError("Invalid port")

        return port


class ServiceType(StrEnum):
    MODEL_MANAGER = auto()
    INFERENCE_SERVICE = auto()
    ...


class ServiceInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: ServiceId
    service_type: ServiceType

    service_endpoint: ServiceEndpoint
