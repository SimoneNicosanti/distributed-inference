from typing import override

from worker.application.contribution.abc.contribution_assembler import (
    ContributionAssembler,
)
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.tensor.tensor_bundle import TensorBundle


class ModelPassResultAssembler(
    ContributionAssembler[ModelPassResultContribution, ModelPassResult]
):
    @override
    async def assemble(
        self,
        contributions: list[ModelPassResultContribution],
    ) -> ModelPassResult:
        if not contributions:
            raise ValueError("Cannot assemble an empty model pass result")

        model_pass_context = contributions[0].model_pass_context
        result_bundle = TensorBundle(bundle={})
        collected_tensor_keys: set[str] = set()
        for contribution in contributions:
            if contribution.model_pass_context != model_pass_context:
                raise ValueError(
                    "Contributions for the same model pass must share the same context"
                )

            tensor_keys = contribution.bundle.get_tensor_names()
            duplicate_tensor_keys = collected_tensor_keys & tensor_keys
            if duplicate_tensor_keys:
                raise ValueError(
                    f"Duplicate model pass result tensors: {duplicate_tensor_keys}"
                )

            collected_tensor_keys.update(tensor_keys)
            result_bundle = result_bundle.merge(contribution.bundle)

        return ModelPassResult(
            context=model_pass_context,
            bundle=result_bundle,
        )
