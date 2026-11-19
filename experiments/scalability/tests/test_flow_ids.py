from uuid import UUID

import pytest
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.flows import flow_pool
from scalability_benchmark.instances import request_record
from scalability_benchmark.randomness import stable_uuid

from shared.flow.flow import FlowId
from shared.service.service import ServiceId


@pytest.mark.parametrize("seed", [1, 42])
def test_flow_uuids_keep_their_seed_mapping_without_an_owner(
    config: ExperimentConfig, seed: int
) -> None:
    device = ServiceId(id="device")
    low_load = flow_pool(config, seed, 0.3, 0.1, device)
    high_load = flow_pool(config, seed, 0.8, 0.1, device)
    expected = tuple(
        FlowId(flow_id=stable_uuid(seed, "flow", str(index)))
        for index in range(config.flows.pool_size)
    )

    assert tuple(flow.flow_id for flow in low_load) == expected
    assert tuple(flow.flow_id for flow in high_load) == expected
    assert len(set(expected)) == len(expected)
    for flow in low_load:
        assert isinstance(flow.flow_id.flow_id, UUID)
        record = request_record(flow)
        assert record["flow_id"] == str(flow.flow_id.flow_id)
        assert "user_id" not in record
