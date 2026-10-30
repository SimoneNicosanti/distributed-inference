# mypy: disable-error-code="attr-defined,name-defined"

from collections.abc import Mapping

import numpy as np
import pyarrow as pa

PROTOCOL_VERSION = 1

SUBMIT_CONTRIBUTION_COMMAND = b"model-pass-result.v1.submit-contribution"
RETRIEVE_RESULT_COMMAND = b"model-pass-result.v1.retrieve"

CONTEXT_FIELD_NAME = "__context_json"
MODEL_PASS_ID_FIELD_NAME = "__model_pass_id"


def build_result_payload_schema(
    tensors: Mapping[str, np.ndarray],
) -> pa.Schema:
    if not tensors:
        raise ValueError("Model pass result payload must contain at least one tensor")

    fields = [
        pa.field(
            CONTEXT_FIELD_NAME,
            pa.large_string(),
            nullable=False,
        )
    ]

    for name, value in tensors.items():
        if not name:
            raise ValueError("Tensor name must not be empty")
        if name in {CONTEXT_FIELD_NAME, MODEL_PASS_ID_FIELD_NAME}:
            raise ValueError(f"Reserved tensor name: {name}")

        array = np.asarray(value)
        fields.append(
            pa.field(
                name,
                pa.fixed_shape_tensor(
                    pa.from_numpy_dtype(array.dtype),
                    list(array.shape),
                ),
                nullable=False,
            )
        )

    return pa.schema(fields)


def validate_result_payload_schema(schema: pa.Schema) -> None:
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Model pass result fields must have unique names")

    context_index = schema.get_field_index(CONTEXT_FIELD_NAME)
    if context_index < 0:
        raise ValueError("Missing model pass result context field")

    context_field = schema.field(context_index)
    if context_field.type != pa.large_string():
        raise ValueError("Model pass result context must be large_string")
    if context_field.nullable:
        raise ValueError("Model pass result context must not be nullable")

    tensor_fields = [field for field in schema if field.name != CONTEXT_FIELD_NAME]
    if not tensor_fields:
        raise ValueError("Model pass result payload must contain at least one tensor")

    for field in tensor_fields:
        if field.name == MODEL_PASS_ID_FIELD_NAME:
            raise ValueError(f"Reserved tensor name: {field.name}")
        if not isinstance(field.type, pa.FixedShapeTensorType):
            raise ValueError(f"Field {field.name} is not a fixed-shape tensor")
        if field.nullable:
            raise ValueError(f"Tensor {field.name} must not be nullable")


def build_retrieve_request_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                MODEL_PASS_ID_FIELD_NAME,
                pa.large_string(),
                nullable=False,
            )
        ]
    )


def validate_retrieve_request_schema(schema: pa.Schema) -> None:
    if schema.names != [MODEL_PASS_ID_FIELD_NAME]:
        raise ValueError("Invalid model pass result retrieval request fields")

    model_pass_id_field = schema.field(MODEL_PASS_ID_FIELD_NAME)
    if model_pass_id_field.type != pa.large_string():
        raise ValueError("Model pass id must be large_string")
    if model_pass_id_field.nullable:
        raise ValueError("Model pass id must not be nullable")
