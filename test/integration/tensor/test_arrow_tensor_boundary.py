import numpy as np

from front_end.adapters.outbound.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from shared.flow.flow import FlowId
from shared.user.user import SYSTEM_USER_ID
from worker.adapters.inbound.ingress.arrow_model_invocation_coordinator import (
    ArrowModelInvocationCoordinator,
)


def test_front_end_and_worker_map_tensor_representations_at_arrow_boundary() -> None:
    flow_id = FlowId(user_id=SYSTEM_USER_ID)
    tensors = {
        "input": np.array([[1.0, 2.0]], dtype=np.float32),
        "mask": np.array([[1, 0]], dtype=np.int64),
    }

    request = ArrowWorkerIngressGateway._encode_request(flow_id, tensors)
    restored_flow_id, worker_bundle = ArrowModelInvocationCoordinator._decode_request(
        request,
        row_index=0,
    )
    response = ArrowModelInvocationCoordinator._encode_result(worker_bundle)
    result = ArrowWorkerIngressGateway._decode_response(response, row_index=0)

    assert restored_flow_id == flow_id
    assert result.bundle.keys() == tensors.keys()
    for name, value in tensors.items():
        np.testing.assert_array_equal(result.bundle[name], value)
        assert result.bundle[name].dtype == value.dtype
