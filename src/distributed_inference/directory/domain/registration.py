from pydantic import BaseModel, ConfigDict

from distributed_inference.directory.domain.service_instance import (
    ServiceInstance,
)
from distributed_inference.domain.identifiers import ServerId


class ServerRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: ServerId


class ServiceRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_instance: ServiceInstance
