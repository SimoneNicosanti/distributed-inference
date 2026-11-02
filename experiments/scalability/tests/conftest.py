from collections.abc import Callable
from pathlib import Path

import pytest
from scalability_benchmark.catalog import CatalogEntry
from scalability_benchmark.configuration import ExperimentConfig, load_configuration
from scalability_benchmark.profiling.static import StaticModel

from controller.adapters.outbound.profile.mappers.model_variant_static_profile_mapper import (
    ModelVariantStaticProfileMapper,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import ProfiledModelVariantDto
from shared.model.model import CNNArchitectureInfo, ModelTask


@pytest.fixture
def config() -> ExperimentConfig:
    return load_configuration(
        Path(__file__).resolve().parents[1] / "config/default.yaml"
    )


@pytest.fixture
def make_model() -> Callable[[CatalogEntry], StaticModel]:
    def create(entry: CatalogEntry) -> StaticModel:
        layers = ("input", "conv", "relu", "pool", "output")
        types = ("Input", "Conv", "Relu", "GlobalAveragePool", "Output")
        tensors = ("a", "b", "c", "d")
        dto = ProfiledModelVariantDto.model_validate(
            {
                "model_variant_id": entry.variant_id,
                "info": {
                    "precision": "fp32",
                    "quantization": "none",
                    "accuracies": [entry.accuracy],
                    "format": "onnx",
                    "inputs_info": {"images": {"shape": [1, 8]}},
                    "dynamic_dimension_values": {},
                },
                "artifact_ref": {"value": "unit-test-artifact"},
                "profile": {
                    "topology": {
                        "layers": {
                            layer: {
                                "name": layer,
                                "type": kind,
                                "is_input": layer == "input",
                                "is_output": layer == "output",
                            }
                            for layer, kind in zip(layers, types, strict=True)
                        },
                        "edges": [
                            {"source": source, "target": target, "tensors": [tensor]}
                            for source, target, tensor in zip(
                                layers[:-1], layers[1:], tensors, strict=True
                            )
                        ],
                    },
                    "invariant_layer_properties": {
                        layer: {
                            "weight_properties": {
                                "weights": {
                                    "shape": [8, 8],
                                    "data_type": "float32",
                                    "data_type_size_in_bytes": 4,
                                }
                            }
                            if layer == "conv"
                            else {}
                        }
                        for layer in layers
                    },
                    "invariant_tensor_properties": {
                        tensor: {
                            "data_type": "float32",
                            "data_type_size_in_bytes": 4,
                            "shape_expression": [1, 8],
                        }
                        for tensor in tensors
                    },
                    "shape_profiles": [
                        {
                            "shape_point": {
                                "input_shape_points": [
                                    {"name": "images", "shape": [1, 8]}
                                ]
                            },
                            "layer_properties": {
                                layer: {
                                    "flops": 1024
                                    if layer == "conv"
                                    else 8
                                    if layer not in {"input", "output"}
                                    else 0
                                }
                                for layer in layers
                            },
                            "tensor_properties": {
                                tensor: {"shape": [1, 8]} for tensor in tensors
                            },
                        }
                    ],
                    "optimization_contractions": [],
                },
            }
        )
        profile = ModelVariantStaticProfileMapper.model_variant_static_profile_from_dto(
            dto,
            ModelDto(
                model_id=entry.variant_id.model_id,
                model_task=ModelTask.CLASSIFICATION,
                architecture_info=CNNArchitectureInfo(),
            ),
        )
        return StaticModel(entry, profile, dto)

    return create
