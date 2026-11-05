"""The compact formulation and its shared operational constraints."""

from controller.adapters.outbound.optimization.compact_bounds import (
    prepare_compact_problem,
)
from controller.adapters.outbound.optimization.compact_execution import (
    add_compact_execution_constraints,
)
from controller.adapters.outbound.optimization.compact_execution_queue import (
    add_compact_execution_queue_constraints,
)
from controller.adapters.outbound.optimization.compact_memory import (
    add_compact_memory_constraints,
)
from controller.adapters.outbound.optimization.compact_network_queue import (
    add_compact_network_queue_constraints,
)
from controller.adapters.outbound.optimization.compact_structure import (
    add_compact_assignment_constraints,
)
from controller.adapters.outbound.optimization.compact_transmission import (
    add_compact_transmission_constraints,
)
from controller.adapters.outbound.optimization.compact_variables import (
    add_compact_variables,
)
from controller.adapters.outbound.optimization.gurobi_qos_constraints import (
    add_response_time_constraints,
)


class CompactFormulation:
    """The sole supported optimization formulation."""

    add_variables = staticmethod(add_compact_variables)
    add_assignment_constraints = staticmethod(add_compact_assignment_constraints)
    add_execution_constraints = staticmethod(add_compact_execution_constraints)
    add_transmission_constraints = staticmethod(add_compact_transmission_constraints)
    add_memory_constraints = staticmethod(add_compact_memory_constraints)
    add_execution_queue_constraints = staticmethod(
        add_compact_execution_queue_constraints
    )
    add_network_queue_constraints = staticmethod(add_compact_network_queue_constraints)
    add_response_time_constraints = staticmethod(add_response_time_constraints)
    prepare_problem = staticmethod(prepare_compact_problem)
