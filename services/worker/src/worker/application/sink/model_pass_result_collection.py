from dataclasses import dataclass

from shared.context.model_pass_context import ModelPassId
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


@dataclass(frozen=True, slots=True)
class ModelPassResultCollection:
    key: ModelPassId
    contributions: tuple[ModelPassResultContribution, ...]
