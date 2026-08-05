import pytest

from distributed_inference.model_manager.domain.model_version import (
    DynamicShapeInfo,
    ShapeType,
    StaticShapeInfo,
)
from distributed_inference.model_manager.domain.model_version_graph import ShapePoint
from distributed_inference.model_profiler.application.ports.outbound.model_graph_extractor import (
    ModelGraphExtractor,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model_version_info,
)


@pytest.mark.unit
def test_only_static_shapes_produce_a_single_shape_point() -> None:
    model_version_info = build_model_version_info(
        static_shapes=[
            StaticShapeInfo(type=ShapeType.BATCH, name="batch_size", value=8),
        ],
        dynamic_shapes=[],
    )

    assert ModelGraphExtractor.compute_shape_points(model_version_info) == [
        ShapePoint(dims=(("batch_size", 8),))
    ]


@pytest.mark.unit
def test_dynamic_shapes_are_expanded_as_the_cartesian_product_of_their_ranges() -> None:
    model_version_info = build_model_version_info(
        static_shapes=[
            StaticShapeInfo(type=ShapeType.BATCH, name="batch_size", value=1),
        ],
        dynamic_shapes=[
            DynamicShapeInfo(
                type=ShapeType.SEQUENCE,
                name="sequence_size",
                min_value=1,
                max_value=5,
                step_size=2,
            ),
        ],
    )

    shape_points = ModelGraphExtractor.compute_shape_points(model_version_info)

    assert shape_points == [
        ShapePoint(dims=(("batch_size", 1), ("sequence_size", 1))),
        ShapePoint(dims=(("batch_size", 1), ("sequence_size", 3))),
        ShapePoint(dims=(("batch_size", 1), ("sequence_size", 5))),
    ]
