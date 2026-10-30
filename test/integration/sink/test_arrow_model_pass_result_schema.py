import numpy as np
import pyarrow as pa
import pytest

from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    CONTEXT_FIELD_NAME,
    MODEL_PASS_ID_FIELD_NAME,
    build_result_payload_schema,
    build_retrieve_request_schema,
    validate_result_payload_schema,
    validate_retrieve_request_schema,
)


def test_builds_valid_result_payload_schema() -> None:
    schema = build_result_payload_schema({"output": np.zeros((2, 3), dtype=np.float32)})

    validate_result_payload_schema(schema)

    assert schema.names == [CONTEXT_FIELD_NAME, "output"]
    assert isinstance(schema.field("output").type, pa.FixedShapeTensorType)


def test_rejects_reserved_tensor_names() -> None:
    with pytest.raises(ValueError, match="Reserved tensor name"):
        build_result_payload_schema(
            {MODEL_PASS_ID_FIELD_NAME: np.zeros((1,), dtype=np.float32)}
        )


def test_builds_valid_retrieve_request_schema() -> None:
    schema = build_retrieve_request_schema()

    validate_retrieve_request_schema(schema)

    assert schema.names == [MODEL_PASS_ID_FIELD_NAME]
