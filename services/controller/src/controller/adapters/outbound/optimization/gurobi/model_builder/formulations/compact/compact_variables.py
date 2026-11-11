from dataclasses import dataclass

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.compute_queue.compute_queue_variables import (
    ComputeQueueVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.response_time.response_time_variables import (
    QoSVariables,
)


@dataclass(slots=True)
class CompactVariables:
    assignment: AssignmentVariables
    execution: ExecutionVariables
    compute_queue: ComputeQueueVariables
    network: NetworkVariables
    memory: MemoryVariables
    qos: QoSVariables
