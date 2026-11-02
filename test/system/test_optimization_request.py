import pytest
from test.system.optimization_scenario import ARRIVAL_RATES, OptimizationScenario


@pytest.mark.integration
def test_optimization_request(optimization_scenario: OptimizationScenario) -> None:
    scenario = optimization_scenario
    assert len(scenario.flow_plans) == len(ARRIVAL_RATES)
    assert len({plan.plan_version for plan in scenario.flow_plans.values()}) == 1
    for plan in scenario.flow_plans.values():
        variant_id = plan.model_specs.model_variant_id
        assert variant_id in scenario.model_variant_ids
        assert plan.ingress_replica_id.model_variant_id == variant_id
        assert plan.sink_replica_id.model_variant_id == variant_id
        assert plan.model_specs.inputs
