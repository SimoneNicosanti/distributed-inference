from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase

import gurobipy as gp
from gurobipy import GRB
from optimizer.application.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)

from controller.domain.backend import Backend
from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId


@dataclass
class _Variable:
    Start: float = 0.25
    LB: float = 0.0
    UB: float = 1.0


class _Model:
    def __init__(self, variables: list[_Variable]) -> None:
        self.variables = variables
        self.update_count = 0

    def getVars(self) -> list[_Variable]:
        return self.variables

    def setAttr(
        self, attribute: str, variables: list[_Variable], values: list[float]
    ) -> None:
        assert attribute == GRB.Attr.Start
        for variable, value in zip(variables, values, strict=True):
            variable.Start = value

    def update(self) -> None:
        self.update_count += 1


class CompactGurobiWarmStartApplierTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.flow = FlowId(UserId())
        self.workers = ServiceId(), ServiceId()
        self.variants = (
            ModelVariantId(ModelId("model"), "a"),
            ModelVariantId(ModelId("model"), "b"),
        )
        self.layers = "input", "hidden", "output"
        self.applier = CompactGurobiWarmStartApplier()
        self._build_target((self.flow,))

    def _build_target(self, flows: tuple[FlowId, ...]) -> None:
        components = range(1, 4)
        self.assignment = SimpleNamespace(
            model_selected={
                (flow, variant): _Variable()
                for flow in flows
                for variant in self.variants
            },
            layer_component={
                (flow, variant, layer, component): _Variable()
                for flow in flows
                for variant in self.variants
                for layer in self.layers
                for component in components
            },
            component_worker={
                (flow, component, worker): _Variable()
                for flow in flows
                for component in components
                for worker in self.workers
            },
            component_active={
                (flow, component): _Variable()
                for flow in flows
                for component in components
            },
            component_last={
                (flow, component): _Variable()
                for flow in flows
                for component in components
            },
        )
        self.priorities = {
            (flow, component, worker, priority): _Variable()
            for flow in flows
            for component in components
            for worker in self.workers
            for priority in (1, 2)
        }
        self.auxiliary = _Variable()
        variables = [
            variable
            for group in (*vars(self.assignment).values(), self.priorities)
            for variable in group.values()
        ]
        self.native = _Model([*variables, self.auxiliary])
        topology = SimpleNamespace(layers_in_topological_order=lambda: self.layers)
        data = SimpleNamespace(
            flows=flows,
            max_components=3,
            static_profiles_by_variant={
                variant: SimpleNamespace(topology=topology) for variant in self.variants
            },
        )
        self.model = GurobiCompiledFormulation(
            native_model=cast(gp.Model, self.native),
            variables=cast(
                CompactVariables,
                SimpleNamespace(
                    assignment=self.assignment,
                    compute_queue=SimpleNamespace(component_priority=self.priorities),
                ),
            ),
            formulation=cast(CompactFormulationData, data),
            construction_time_s=0.0,
        )

    def _solution(
        self,
        *,
        flow: FlowId | None = None,
        variant: ModelVariantId | None = None,
        layers: tuple[tuple[str, ...], ...] = (("input", "hidden"), ("output",)),
        workers: tuple[WorkerId, ...] | None = None,
        priorities: tuple[int, ...] | None = None,
    ) -> OptimizationSolution:
        workers = workers or self.workers
        priorities = priorities or (1,) * len(layers)
        components = tuple(
            ComponentSolution(
                component_index=index,
                worker_id=worker,
                backend=Backend.CPU,
                layer_keys=keys,
                input_layer_keys=tuple(key for key in keys if key == "input"),
                output_layer_keys=tuple(key for key in keys if key == "output"),
                tensor_contract=ComponentTensorContract((), ()),
                priority=priority,
                mean_execution_time_s=0.0,
                mean_execution_queue_time_s=0.0,
                mean_transmission_time_s=0.0,
                transmission_latency_s=0.0,
            )
            for index, (keys, worker, priority) in enumerate(
                zip(layers, workers, priorities, strict=True), start=1
            )
        )
        flow_solution = FlowOptimizationSolution(
            flow_id=flow or self.flow,
            ingress_worker_id=components[0].worker_id,
            sink_worker_id=components[-1].worker_id,
            model_variant_id=variant or self.variants[0],
            components=components,
            cuts=(),
            mean_batch_fill_time_s=0.0,
            mean_response_time_s=0.0,
        )
        return OptimizationSolution((flow_solution,), 0.0, False)

    async def test_transfers_reduced_seed_with_undefined_auxiliaries(self) -> None:
        bounds = [(variable.LB, variable.UB) for variable in self.native.variables]
        await self.applier.apply_warm_start(self.model, self._solution())

        for (_, variant), variable in self.assignment.model_selected.items():
            self.assertEqual(variable.Start, float(variant == self.variants[0]))
        for key, variable in self.assignment.layer_component.items():
            _, variant, layer, component = key
            expected = variant == self.variants[0] and component == (
                2 if layer == "output" else 1
            )
            self.assertEqual(variable.Start, float(expected))
        for key, variable in self.assignment.component_worker.items():
            _, component, worker = key
            expected = component <= 2 and worker == self.workers[component - 1]
            self.assertEqual(variable.Start, float(expected))
        for (_, component), variable in self.assignment.component_active.items():
            self.assertEqual(variable.Start, float(component <= 2))
        for (_, component), variable in self.assignment.component_last.items():
            self.assertEqual(variable.Start, float(component == 2))
        for (_, component, worker, priority), variable in self.priorities.items():
            expected = (
                component <= 2
                and worker == self.workers[component - 1]
                and priority == 1
            )
            self.assertEqual(variable.Start, float(expected))
        self.assertEqual(self.auxiliary.Start, GRB.UNDEFINED)
        self.assertEqual(bounds, [(v.LB, v.UB) for v in self.native.variables])
        self.assertEqual(self.native.update_count, 1)

    async def test_reapplication_replaces_previous_decisions(self) -> None:
        first = self._solution(
            layers=(("input",), ("hidden",), ("output",)),
            workers=(*self.workers, self.workers[0]),
            priorities=(1, 1, 2),
        )
        await self.applier.apply_warm_start(self.model, first)
        second = self._solution(variant=self.variants[1], workers=self.workers[::-1])
        await self.applier.apply_warm_start(self.model, second)

        self.assertEqual(
            self.assignment.model_selected[self.flow, self.variants[0]].Start, 0
        )
        self.assertEqual(
            self.assignment.model_selected[self.flow, self.variants[1]].Start, 1
        )
        self.assertEqual(self.assignment.component_active[self.flow, 3].Start, 0)
        self.assertEqual(self.assignment.component_last[self.flow, 3].Start, 0)
        self.assertEqual(self.assignment.component_last[self.flow, 2].Start, 1)
        self.assertEqual(self.priorities[self.flow, 3, self.workers[0], 2].Start, 0)
        self.assertEqual(
            self.assignment.component_worker[self.flow, 1, self.workers[1]].Start, 1
        )
        self.assertEqual(self.native.update_count, 2)

    async def test_maps_multiple_flows_independently(self) -> None:
        second_flow = FlowId(UserId())
        self._build_target((self.flow, second_flow))
        first = self._solution()
        second = self._solution(flow=second_flow, variant=self.variants[1])
        seed = OptimizationSolution(
            first.flow_solutions + second.flow_solutions, 0, False
        )
        await self.applier.apply_warm_start(self.model, seed)

        for (flow, variant), variable in self.assignment.model_selected.items():
            expected = self.variants[0] if flow == self.flow else self.variants[1]
            self.assertEqual(variable.Start, float(variant == expected))

    async def test_invalid_seeds_are_rejected_before_changing_starts(self) -> None:
        seed = self._solution()
        flow_solution = seed.flow_solutions[0]
        gap = replace(flow_solution.components[1], component_index=3)
        invalid = (
            replace(seed, flow_solutions=()),
            self._solution(flow=FlowId(UserId())),
            self._solution(variant=ModelVariantId(ModelId("unknown"), "a")),
            self._solution(workers=(ServiceId(), self.workers[1])),
            self._solution(priorities=(1, 3)),
            self._solution(
                layers=(("input",), ("hidden",), (), ("output",)),
                workers=(*self.workers, *self.workers),
            ),
            self._solution(layers=(("input", "hidden"), ("hidden", "output"))),
            self._solution(layers=(("input", "hidden", "unknown"), ("output",))),
            self._solution(layers=(("hidden",), ("output",))),
            replace(
                seed,
                flow_solutions=(
                    replace(
                        flow_solution,
                        components=(flow_solution.components[0], gap),
                    ),
                ),
            ),
        )
        for index, candidate in enumerate(invalid):
            with self.subTest(candidate=index):
                with self.assertRaises(ValueError):
                    await self.applier.apply_warm_start(self.model, candidate)
                self.assertTrue(all(v.Start == 0.25 for v in self.native.variables))
                self.assertEqual(self.native.update_count, 0)

    async def test_invalid_later_flow_preserves_all_existing_starts(self) -> None:
        second_flow = FlowId(UserId())
        self._build_target((self.flow, second_flow))
        first = self._solution()
        second = self._solution(flow=second_flow, priorities=(1, 3))
        seed = OptimizationSolution(
            first.flow_solutions + second.flow_solutions, 0, False
        )
        with self.assertRaisesRegex(ValueError, "priority"):
            await self.applier.apply_warm_start(self.model, seed)
        self.assertTrue(all(v.Start == 0.25 for v in self.native.variables))
        self.assertEqual(self.native.update_count, 0)
