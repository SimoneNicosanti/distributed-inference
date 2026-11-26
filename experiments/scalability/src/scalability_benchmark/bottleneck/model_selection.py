"""Predeclare each flow's variant without consulting any optimizer solution."""

from typing import Any

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ModelConfigurationKey,
)
from scalability_benchmark.bottleneck.instances import jsonable
from scalability_benchmark.randomness import stream
from shared.flow.flow import FlowId

REFERENCE_POLICY = "predefined_models_from_b_v1"
type ModelChoices = dict[FlowId, ModelConfigurationKey]


def predefined_models(problem: CompactOptimizationProblem, seed: int) -> ModelChoices:
    # Same addressable namespace as the existing scalability model_variant mode.
    # The benchmark has one input configuration per variant; reject ambiguous
    # variants rather than silently fixing an additional batch/shape decision.
    choices = {}
    for flow, data in problem.flows.items():
        if not data.configurations or len(
            {c.model_variant_id for c in data.configurations}
        ) != len(data.configurations):
            raise ValueError(
                "Predefined variants require one eligible configuration per variant"
            )
        choices[flow] = stream(seed, "model-selection", str(flow.flow_id)).choice(
            data.configurations
        )
    return choices


def assignment_record(choices: ModelChoices) -> list[dict[str, Any]]:
    return [
        {"flow_id": str(flow.flow_id), "configuration": jsonable(configuration)}
        for flow, configuration in sorted(
            choices.items(), key=lambda item: str(item[0].flow_id)
        )
    ]
