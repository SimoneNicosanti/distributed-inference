from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class UserId:
    id: UUID = field(default_factory=uuid4)


SYSTEM_USER_ID = UserId(id=UUID("8db917c1-2494-4b25-a79c-12f97cb67942"))
