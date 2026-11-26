import asyncio
from pathlib import Path

import onnx
import pytest
import yaml
from onnx import TensorProto, helper
from priority_effect.configuration import Config, load_config
from priority_effect.scenarios import prepare


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> Config:
    root = Path(__file__).resolve().parents[1]
    temporary = tmp_path_factory.mktemp("priority-models")
    settings = load_config(root / "config/default.yaml")
    base = yaml.safe_load(settings.base_config.read_text())
    base["root"] = str(settings.base_config.parent.parent)
    base["catalog"]["directory"] = str(temporary)
    for specification in base["catalog"]["models"]:
        specification["input_shape"] = [1, 3, 4, 4]
    base_path = temporary / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    for name, count in (("resnet50", 1), ("efficientnet_b0", 3)):
        nodes = [
            helper.make_node(
                "Relu",
                ["images" if i == 0 else f"t{i}"],
                [f"t{i + 1}"],
                name=f"relu{i}",
            )
            for i in range(count)
        ]
        graph = helper.make_graph(
            nodes,
            name,
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])],
            [
                helper.make_tensor_value_info(
                    f"t{count}", TensorProto.FLOAT, [1, 3, 4, 4]
                )
            ],
        )
        onnx.save(
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
            ),
            temporary / f"{name}.onnx",
        )
    raw = settings.model_dump()
    raw.update(
        {
            "base_config": base_path,
            "output": temporary / "output",
            "flows": [2],
            "target_loads": [0.3, 0.8],
            "pilot_loads": [0.3, 0.8],
        }
    )
    raw["workloads"][1]["models"] = ("resnet50", "efficientnet_b0")
    # Toy hardware is independent of the realistic campaign defaults.
    raw["hardware"]["edge"]["overhead_s"] = 1.0
    raw["hardware"]["device"]["overhead_s"] = 2.0
    return Config.model_validate(raw)


@pytest.fixture(scope="module")
def prepared(config, tmp_path_factory):
    return asyncio.run(
        prepare(config, 42, tmp_path_factory.mktemp("profiles"), "unit-fixture")
    )
