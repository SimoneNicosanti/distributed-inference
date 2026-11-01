from front_end.domain.flow_plan import FlowPlan
from front_end.domain.model_specs import InputSpec, ModelSpecs
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto


class FlowPlanDtoMapper:
    @staticmethod
    def to_flow_plan(dto: FlowPlanDto) -> FlowPlan:
        return FlowPlan(
            plan_version=dto.plan_version,
            model_specs=ModelSpecs(
                model_variant_id=dto.model_specs.model_variant_id,
                inputs=tuple(
                    InputSpec(name=spec.name, shape=spec.shape, dtype=spec.dtype)
                    for spec in dto.model_specs.inputs
                ),
            ),
            ingress_replica_id=dto.ingress_replica_id,
            sink_replica_id=dto.sink_replica_id,
        )
