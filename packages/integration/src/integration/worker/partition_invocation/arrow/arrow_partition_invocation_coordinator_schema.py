from collections.abc import Mapping

import numpy as np
import pyarrow as pa

PROTOCOL_VERSION = 1

SUBMIT_CONTRIBUTION_COMMAND = b"partition-invocation.v1.submit-contribution"

CONTEXT_FIELD_NAME = "__context_json"

## The schema changes depending on the number of tensors
## First column is the context
## All other columns are the tensors


def build_contribution_schema(
    tensors: Mapping[str, np.ndarray],
) -> pa.Schema:
    if not tensors:
        raise ValueError("Contribution must contain at least one tensor")

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

        if name == CONTEXT_FIELD_NAME:
            raise ValueError(f"Reserved tensor name: {name}")

        value = np.asarray(value)

        tensor_type = pa.fixed_shape_tensor(
            pa.from_numpy_dtype(value.dtype),
            list(value.shape),
        )

        fields.append(
            pa.field(
                name,
                tensor_type,
                nullable=False,
            )
        )

    return pa.schema(fields)


def validate_contribution_schema(
    schema: pa.Schema,
) -> None:
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Contribution fields must have unique names")

    context_index = schema.get_field_index(CONTEXT_FIELD_NAME)

    if context_index < 0:
        raise ValueError("Missing contribution context field")

    context_field = schema.field(context_index)

    if context_field.type != pa.large_string():
        raise ValueError("Contribution context must be large_string")

    if context_field.nullable:
        raise ValueError("Contribution context must not be nullable")

    tensor_fields = [field for field in schema if field.name != CONTEXT_FIELD_NAME]

    if not tensor_fields:
        raise ValueError("Contribution must contain at least one tensor")

    for field in tensor_fields:
        if not isinstance(
            field.type,
            pa.FixedShapeTensorType,
        ):
            raise ValueError(f"Field {field.name} is not a fixed-shape tensor")

        if field.nullable:
            raise ValueError(f"Tensor {field.name} must not be nullable")
