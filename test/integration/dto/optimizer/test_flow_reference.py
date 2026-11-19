import json
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from integration.dto.optimizer.common import FlowReference
from integration.shared.id_json import flow_id_from_json, flow_id_to_json
from shared.flow.flow import FlowId


def test_flow_reference_round_trips_only_the_flow_uuid() -> None:
    identifier = UUID("00000000-0000-0000-0000-000000000002")
    flow = FlowId(flow_id=identifier)
    adapter = TypeAdapter(FlowReference)

    payload = flow_id_to_json(flow)

    assert json.loads(payload) == {"flow_id": str(identifier)}
    assert flow_id_from_json(payload) == flow
    assert adapter.validate_json(payload) == flow
    assert json.loads(adapter.dump_json(flow)) == json.loads(payload)
    assert "user_id" not in json.dumps(adapter.json_schema())


@pytest.mark.parametrize(
    "payload",
    [{}, {"flow_id": None}, {"flow_id": "not-a-uuid"}, {"flow_id": 2}],
)
def test_flow_reference_requires_an_explicit_valid_uuid(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(FlowReference).validate_python(payload)
