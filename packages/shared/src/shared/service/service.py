# We can have multiple users.
# - Each user can define multiple flows.
# - Each flow can specify the model (or task type) to be executed.
# - Each model has multiple versions.
# - Each model version can be divided in multiple components after the optimization
# - Then we have the artifacts as stored in the model store.
import os
from dataclasses import dataclass, field
from uuid import UUID, uuid4


def service_id_factory() -> str | UUID:
    service_id = os.getenv("SERVICE_ID", None)
    if service_id:
        return service_id
    return uuid4()


@dataclass(frozen=True, slots=True)
class ServiceId:
    id: str | UUID = field(default_factory=service_id_factory)


type WorkerId = ServiceId

type IngressId = ServiceId
type SinkId = ServiceId
