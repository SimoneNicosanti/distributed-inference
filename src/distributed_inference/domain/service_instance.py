from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import ServiceId


class ServiceEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    protocol: str


class ServiceType(StrEnum):
    MODEL_MANAGER = auto()
    INFERENCE_SERVICE = auto()
    ...


class ServiceInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: ServiceId
    service_type: ServiceType

    service_endpoint: ServiceEndpoint
