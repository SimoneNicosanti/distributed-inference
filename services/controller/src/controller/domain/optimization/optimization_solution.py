from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class OptimizationSolutionId(BaseModel):
    model_config = ConfigDict(frozen=True)


class ComponentSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    component_index: PositiveInt
    worker_id: WorkerId
    layer_keys: tuple[LayerKey, ...]
    priority: PositiveInt

    mean_execution_time_s: NonNegativeFloat
    mean_execution_queue_time_s: NonNegativeFloat
    mean_transmission_time_s: NonNegativeFloat
    transmission_latency_s: NonNegativeFloat


class FlowOptimizationSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    model_variant_id: ModelVariantId
    components: tuple[ComponentSolution, ...]

    mean_batch_fill_time_s: NonNegativeFloat
    mean_response_time_s: NonNegativeFloat


class OptimizationSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_solutions: tuple[FlowOptimizationSolution, ...]
    objective_value: float
    is_optimal: bool
