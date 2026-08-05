import pytest
from pydantic import ValidationError

from distributed_inference.model_manager.domain.sub_model import SubModelId
from test.support.model_manager.model_domain_test_utils import build_model_version_id


@pytest.mark.unit
def test_sub_model_id_canonicalizes_layer_order() -> None:
    model_version_id = build_model_version_id()

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
    with pytest.raises(ValidationError, match="layers must not contain duplicates"):
        SubModelId(
            model_version_id=build_model_version_id(),
            layers=("encoder.0", "encoder.0"),
        )


@pytest.mark.unit
def test_sub_model_id_rejects_empty_layers() -> None:
    with pytest.raises(ValidationError, match="layers must not be empty"):
        SubModelId(model_version_id=build_model_version_id(), layers=())


@pytest.mark.unit
def test_sub_model_id_rejects_string_layer_collection() -> None:
    with pytest.raises(ValueError, match="Layers must contain layer names"):
        SubModelId.check_valid_layers_format("encoder.0")


@pytest.mark.unit
def test_sub_model_id_is_immutable() -> None:
    sub_model_id = SubModelId(
        model_version_id=build_model_version_id(),
        layers=("encoder.0",),
    )

    with pytest.raises(ValidationError):
        sub_model_id.layers = ("encoder.1",)
