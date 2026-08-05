import pytest
from pydantic import ValidationError

from distributed_inference.model_manager.domain.model_version import (
    DynamicShapeInfo,
    ShapeType,
    StaticShapeInfo,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model_id,
    build_model_version,
    build_model_version_id,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("min_value", "max_value", "step_size", "message"),
    [
        (0, 8, 1, "min must be > 0"),
        (1, 0, 1, "max must be > 0"),
        (1, 8, 0, "steps must be > 0"),
        (8, 8, 1, "min must be < max"),
    ],
)
def test_dynamic_shape_info_rejects_inconsistent_ranges(
    min_value: int,
    max_value: int,
    step_size: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        DynamicShapeInfo(
            type=ShapeType.BATCH,
            name="batch_size",
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
        )


@pytest.mark.unit
def test_static_shape_info_rejects_non_positive_values() -> None:
    with pytest.raises(ValidationError, match="value must be > 0"):
        StaticShapeInfo(type=ShapeType.SEQUENCE, name="sequence_size", value=0)


@pytest.mark.unit
def test_model_version_exposes_the_model_id_of_its_version_id() -> None:
    model_id = build_model_id()
    model_version = build_model_version(
        model_version_id=build_model_version_id(model_id=model_id, version_tag="v2")
    )

    assert model_version.model_id == model_id
    assert model_version.model_version_id.version_tag == "v2"
