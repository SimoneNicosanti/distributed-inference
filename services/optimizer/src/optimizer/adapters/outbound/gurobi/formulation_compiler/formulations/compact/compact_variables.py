from dataclasses import dataclass

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.compute_queue.compute_queue_variables import (
    ComputeQueueVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.response_time.response_time_variables import (
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
