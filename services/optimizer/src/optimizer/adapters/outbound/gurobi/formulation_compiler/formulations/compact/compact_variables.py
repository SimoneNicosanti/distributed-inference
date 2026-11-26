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

    def extend(self, added: CompactVariables) -> None:
        """Append flow-local maps; builders reuse the worker-level maps."""
        self.assignment.extend(added.assignment)
        self.execution.execution_time_s.update(added.execution.execution_time_s)
        queue, new_queue = self.compute_queue, added.compute_queue
        queue.component_priority.update(new_queue.component_priority)
        queue.priority_class_utilization.update(new_queue.priority_class_utilization)
        queue.component_queue_wait_s.update(new_queue.component_queue_wait_s)
        network, new_network = self.network, added.network
        network.tensor_crossing_auxiliary.update(new_network.tensor_crossing_auxiliary)
        network.source_latency_s.update(new_network.source_latency_s)
        network.transmission_time_s.update(new_network.transmission_time_s)
        network.component_transmission_response_s.update(
            new_network.component_transmission_response_s
        )
        network.source_response_s.update(new_network.source_response_s)
        network.nominal_upper_s.update(new_network.nominal_upper_s)
        self.memory.component_weight_mb.update(added.memory.component_weight_mb)
        self.qos.response_time_s.update(added.qos.response_time_s)
        self.qos.response_delay_s.update(added.qos.response_delay_s)
        self.qos.deadline_violation.update(added.qos.deadline_violation)
