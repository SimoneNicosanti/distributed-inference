"""Small normalized instances without importing any solver SDK."""

from uuid import UUID

from optimizer.application.evaluation.compact_solution_evaluator import FlowDecisions
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    FlowData,
    ModelConfigurationKey,
    ObjectiveKind,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.local_worker_config import ValidLocalWorkerConfig
from optimizer.domain.problem.other.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import ServiceId


def problem(
    *,
    flows=1,
    variants=2,
    workers=2,
    components=3,
    priorities=2,
    rate=0.2,
    deadline=10.0,
    objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
    batch=1,
    zero_service=False,
):
    hosts = tuple(ServiceId(UUID(int=100 + i)) for i in range(workers))
    configs, graphs, times = {}, {}, {}
    for index in range(variants):
        key = ModelConfigurationKey(
            ModelVariantId(ModelId(f"model-{index}"), "fp32"),
            ModelInputConcreteShapes({"t0": (batch, 3)}),
        )
        layers = (
            ("input", "compute", "output")
            if index == 0
            else ("input", "a", "b", "join", "output")
        )
        edges = (
            (("input", "compute", "t0"), ("compute", "output", "t3"))
            if index == 0
            else (
                ("input", "a", "t0"),
                ("input", "b", "t0"),
                ("a", "join", "t1"),
                ("b", "join", "t2"),
                ("join", "output", "t3"),
            )
        )
        metric = PerformanceMetric(
            PerformanceMetricType.ACCURACY, "test", 0.8 + 0.1 * index
        )
        graphs[key.model_variant_id] = VariantData(
            (metric,),
            layers,
            ("input",),
            ("output",),
            tuple(
                EdgeDescription(s, t, frozenset((tensor,))) for s, t, tensor in edges
            ),
            {layer: 0 if layer in ("input", "output") else 128 for layer in layers},
        )
        sizes = {
            "t0": 100 * batch,
            "t1": 70 * batch,
            "t2": 60 * batch,
            "t3": 20 * batch,
        }
        configs[key] = ConfigurationData(batch, sizes, 500 * batch)
        for i, host in enumerate(hosts):
            costs = {
                "input": 0.0,
                "output": 0.0,
                "compute": 0.03 if i == 0 else 0.006,
                "a": 0.04 if i == 0 else 0.005,
                "b": 0.002 if i == 0 else 0.015,
                "join": 0.01 if i == 0 else 0.005,
            }
            times[key, host] = {
                layer: 0.0 if zero_service else costs[layer] * batch for layer in layers
            }
    requirement = PerformanceMetric(PerformanceMetricType.ACCURACY, "test", 0.0)
    return CompactOptimizationProblem(
        graphs,
        configs,
        times,
        {
            FlowId(flow_id=UUID(int=1 + i)): FlowData(
                rate,
                deadline,
                requirement,
                1e6,
                tuple(configs),
                hosts,
                FlowEndpointPlacement(
                    FixedPlacement(hosts[0]), FixedPlacement(hosts[0])
                ),
                ValidLocalWorkerConfig(hosts[0], True),
            )
            for i in range(flows)
        },
        {
            host: WorkerData(Backend.CPU if i == 0 else Backend.GPU, 1_000_000_000)
            for i, host in enumerate(hosts)
        },
        tuple(
            NetworkConnection(s, t, 100_000.0, 0.002)
            for s in hosts
            for t in hosts
            if s != t
        ),
        OptimizationPolicy(components, priorities, 0.8, 0.7, objective),
    )


def decisions(instance, *, split=False, variant=0):
    key = tuple(instance.configurations)[variant]
    graph = instance.variants[key.model_variant_id]
    hosts = tuple(instance.workers)
    return {
        flow: FlowDecisions(
            key,
            {
                layer: (1 if layer == "input" else 3 if layer == "output" else 2)
                if split
                else 1
                for layer in graph.layers
            },
            (hosts[0], hosts[1], hosts[0]) if split else (hosts[0],),
            (1, 1, 1) if split else (1,),
        )
        for flow in instance.flows
    }
