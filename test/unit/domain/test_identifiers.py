from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from distributed_inference.domain.identifiers import (
    SYSTEM_USER_ID,
    ServerId,
    ServiceId,
    UserId,
)


@pytest.mark.unit
def test_identifiers_are_frozen_and_hashable() -> None:
    server_id = ServerId(id=uuid4())
    service_id = ServiceId(server_id=server_id, service_id=uuid4())

    assert {service_id, service_id.model_copy()} == {service_id}

    with pytest.raises(ValidationError):
        service_id.server_id = ServerId(id=uuid4())


@pytest.mark.unit
def test_identifiers_generate_a_distinct_id_by_default() -> None:
    assert UserId().id != UserId().id
    assert ServerId().id != ServerId().id


@pytest.mark.unit
def test_system_user_id_is_a_stable_well_known_value() -> None:
    assert SYSTEM_USER_ID == UserId(id=UUID("8db917c1-2494-4b25-a79c-12f97cb67942"))
