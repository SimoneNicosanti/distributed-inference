from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.service_instance import ServiceInstance


class ServiceRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_instance: ServiceInstance
