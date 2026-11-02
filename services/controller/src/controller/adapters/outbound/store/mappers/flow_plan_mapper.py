from controller.domain.plan.front_end_plan import FlowPlan
from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)


class FlowPlanMapper:
    @staticmethod
    def to_flow_plan_dto(plan: FlowPlan) -> FlowPlanDto:
        return FlowPlanDto(
            plan_version=plan.plan_version,
            model_specs=ModelSpecsDto(
                model_variant_id=plan.model_variant_id,
                inputs=tuple(
                    InputSpecDto(
                        name=tensor_key,
                        shape=property_.shape_expression,
                        dtype=property_.data_type,
                    )
                    for tensor_key, property_ in sorted(
                        plan.input_tensor_properties.items()
                    )
                ),
            ),
            ingress_replica_id=plan.ingress_replica_id,
            sink_replica_id=plan.sink_replica_id,
        )
