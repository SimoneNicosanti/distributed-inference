from uuid import uuid4

import pytest
from pydantic import ValidationError

from distributed_inference.domain.identifiers import (
    FlowId,
    ModelId,
    ModelVersionId,
    RequestId,
    SubModelId,
    UserId,
)


def _model_version_id() -> ModelVersionId:
    user_id = UserId(user_id=uuid4())
    model_id = ModelId(user_id=user_id, model_name="vision-model")
    return ModelVersionId(model_id=model_id, version_number=3)


@pytest.mark.unit
def test_sub_model_id_canonicalizes_layer_order() -> None:
    model_version_id = _model_version_id()

    first = SubModelId(
        model_version_id=model_version_id,
        layers=("encoder.2", "encoder.0", "encoder.1"),
    )
    second = SubModelId(
        model_version_id=model_version_id,
        layers=("encoder.1", "encoder.2", "encoder.0"),
    )

    assert first.layers == ("encoder.0", "encoder.1", "encoder.2")
    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.unit
def test_sub_model_id_rejects_duplicate_layers() -> None:
    with pytest.raises(
        ValidationError,
        match="layers must not contain duplicates",
    ):
        SubModelId(
            model_version_id=_model_version_id(),
            layers=("encoder.0", "encoder.0"),
        )


@pytest.mark.unit
def test_nested_identifiers_are_immutable() -> None:
    model_version_id = _model_version_id()

    with pytest.raises(ValidationError):
        model_version_id.version_number = 4


@pytest.mark.unit
def test_sub_model_id_rejects_empty_layers() -> None:
    with pytest.raises(ValidationError, match="layers must not be empty"):
        SubModelId(
            model_version_id=_model_version_id(),
            layers=(),
        )


@pytest.mark.unit
def test_sub_model_id_rejects_string_layer_collection() -> None:
    with pytest.raises(ValueError, match="Layers must contain layer names"):
        SubModelId.check_valid_layers_format("encoder.0")


@pytest.mark.unit
def test_request_id_requires_explicit_request_index() -> None:
    model_version_id = _model_version_id()
    flow_id = FlowId(
        user_id=model_version_id.model_id.user_id,
        flow_id=uuid4(),
    )
    sub_model_id = SubModelId(
        model_version_id=model_version_id,
        layers=("encoder.0",),
    )

    with pytest.raises(ValidationError, match="request_idx"):
        RequestId.model_validate(
            {
                "flow_id": flow_id,
                "sub_model_id": sub_model_id,
            }
        )
