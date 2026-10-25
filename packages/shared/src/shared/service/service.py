from dataclasses import dataclass, field
from uuid import UUID, uuid4

# We can have multiple users.
# - Each user can define multiple flows.
# - Each flow can specify the model (or task type) to be executed.
# - Each model has multiple versions.
# - Each model version can be divided in multiple components after the optimization
# - Then we have the artifacts as stored in the model store.


@dataclass(frozen=True, slots=True)
class ServiceId:
    id: UUID = field(default_factory=uuid4)


type WorkerId = ServiceId

type ResultSinkId = ServiceId
