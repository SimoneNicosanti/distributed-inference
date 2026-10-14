# mypy: disable-error-code="attr-defined,name-defined"

from collections.abc import Mapping

import numpy as np
import pyarrow as pa

RUN_MODEL_INVOCATION_COMMAND = b"model-invocation.v1.run"

FLOW_ID_FIELD_NAME = "__flow_id_json"


def build_model_invocation_request_schema(
    tensors: Mapping[str, np.ndarray],
) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                FLOW_ID_FIELD_NAME,
                pa.large_string(),
                nullable=False,
            ),
            *_build_tensor_fields(tensors),
        ]
    )


def build_model_invocation_result_schema(
    tensors: Mapping[str, np.ndarray],
) -> pa.Schema:
    return pa.schema(_build_tensor_fields(tensors))


def _build_tensor_fields(
    tensors: Mapping[str, np.ndarray],
) -> list[pa.Field]:
    if not tensors:
        raise ValueError("Model invocation must contain at least one tensor")

    fields: list[pa.Field] = []

    for name, value in tensors.items():
        if not name:
            raise ValueError("Tensor name must not be empty")

        if name == FLOW_ID_FIELD_NAME:
            raise ValueError(f"Reserved tensor name: {name}")

        value = np.asarray(value)
        fields.append(
            pa.field(
                name,
                pa.fixed_shape_tensor(
                    pa.from_numpy_dtype(value.dtype),
                    list(value.shape),
                ),
                nullable=False,
            )
        )

    return fields


def validate_model_invocation_request_schema(schema: pa.Schema) -> None:
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Model invocation fields must have unique names")

    flow_id_index = schema.get_field_index(FLOW_ID_FIELD_NAME)
    if flow_id_index < 0:
        raise ValueError("Missing model invocation flow id field")

    flow_id_field = schema.field(flow_id_index)
    if flow_id_field.type != pa.large_string():
        raise ValueError("Model invocation flow id must be large_string")

    if flow_id_field.nullable:
        raise ValueError("Model invocation flow id must not be nullable")

    tensor_fields = [field for field in schema if field.name != FLOW_ID_FIELD_NAME]
    _validate_tensor_fields(tensor_fields)


def validate_model_invocation_result_schema(schema: pa.Schema) -> None:
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Model invocation fields must have unique names")

    if FLOW_ID_FIELD_NAME in schema.names:
        raise ValueError(
            f"Reserved model invocation result field: {FLOW_ID_FIELD_NAME}"
        )

    _validate_tensor_fields(list(schema))


def _validate_tensor_fields(tensor_fields: list[pa.Field]) -> None:
    if not tensor_fields:
        raise ValueError("Model invocation must contain at least one tensor")

    for field in tensor_fields:
        if not isinstance(field.type, pa.FixedShapeTensorType):
            raise ValueError(f"Field {field.name} is not a fixed-shape tensor")

        if field.nullable:
            raise ValueError(f"Tensor {field.name} must not be nullable")
