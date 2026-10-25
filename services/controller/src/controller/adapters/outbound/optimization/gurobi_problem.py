from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId

type FlowModel = tuple[FlowId, ModelVariantId]
type ModelLayer = tuple[ModelVariantId, LayerKey]
type ModelTensor = tuple[ModelVariantId, TensorKey]


@dataclass(frozen=True, slots=True)
class GurobiProblem:
    # Ordered and filtered index sets used by the formulation.
    flows: tuple[FlowId, ...]
    workers: tuple[WorkerId, ...]
    workers_by_flow: dict[FlowId, tuple[WorkerId, ...]]
    models_by_flow: dict[FlowId, tuple[ModelVariantId, ...]]

    # Coefficients derived from multiple domain inputs.
    service_arrival_rate_per_s: dict[FlowModel, float]
    batch_fill_time_s: dict[FlowModel, float]
    compute_bin_limits: tuple[float, ...]
    network_bin_limits: tuple[float, ...]
    max_queue_wait_s_by_worker: dict[WorkerId, float]
    max_queue_wait_s_by_flow: dict[FlowId, float]

    # Stable numeric names for Gurobi variables and constraints.
    flow_index: dict[FlowId, int]
    worker_index: dict[WorkerId, int]
    model_index: dict[FlowModel, int]
    layer_index: dict[ModelLayer, int]
    tensor_index: dict[ModelTensor, int]
