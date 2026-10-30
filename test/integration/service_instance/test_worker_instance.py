import pytest
from pydantic import ValidationError

from integration.service_instance.service_instance import Capability, CapabilityType
from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import ServiceId


def _capability(capability_type: CapabilityType) -> Capability:
    return Capability(type=capability_type, interfaces=[])


def test_worker_requires_result_retrieve_capability() -> None:
    with pytest.raises(ValidationError, match="result_retrieve"):
        WorkerInstance(
            service_id=ServiceId(),
            capabilities=[
                _capability(CapabilityType.HOP_INVOCATION),
                _capability(CapabilityType.PLAN_ACTUATION),
                _capability(CapabilityType.IN_NETWORK_PROFILING),
            ],
        )


def test_worker_accepts_all_required_capabilities() -> None:
    worker = WorkerInstance(
        service_id=ServiceId(),
        capabilities=[
            _capability(CapabilityType.HOP_INVOCATION),
            _capability(CapabilityType.PLAN_ACTUATION),
            _capability(CapabilityType.IN_NETWORK_PROFILING),
            _capability(CapabilityType.RESULT_RETRIEVE),
        ],
    )

    assert worker.get_capabilities_by_type(CapabilityType.RESULT_RETRIEVE)
