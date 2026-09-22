from pydantic import BaseModel, ConfigDict

from integration.directory.service_instance import (
    ServiceInstance,
)
from integration.identifiers.identifiers import ServerId


class ServerRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: ServerId


class ServiceRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_instance: ServiceInstance
