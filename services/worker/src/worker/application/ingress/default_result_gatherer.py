import asyncio
from dataclasses import dataclass
from typing import override

from shared.context.model_pass_context import ModelPassId
from shared.model.keys import TensorKey
from shared.tensor.tensor_bundle import TensorBundle
from worker.application.ingress.abc.result_waiter import ResultWaiter
from worker.application.ports.inbound.ingress.result_contribution_sink import (
    ResultContributionSink,
)
from worker.domain.model_pass.model_pass_error import ModelPassError
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class DefaultResultGatherer(ResultContributionSink, ResultWaiter):
    @dataclass
    class _PendingResult:
        future: asyncio.Future[TensorBundle]
        expected_tensor_keys: frozenset[TensorKey]
        bundle: TensorBundle

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._pending_results: dict[
            ModelPassId, DefaultResultGatherer._PendingResult
        ] = {}

    @override
    async def sink_contribution(
        self, contribution: ModelPassResultContribution
    ) -> None:
        model_pass_id = contribution.model_pass_id

        pending_result = self._pending_results.get(model_pass_id)
        if pending_result is None:
            raise ValueError(f"No pending result for model pass {model_pass_id}")

        if pending_result.future.done():
            self._pending_results.pop(model_pass_id)
            raise ValueError(
                f"Result wait already completed for model pass {model_pass_id}"
            )

        pending_result.bundle = pending_result.bundle.merge(contribution.bundle)

        if (
            pending_result.expected_tensor_keys
            <= pending_result.bundle.get_tensor_names()
        ):
            result = pending_result.bundle.filter(
                list(pending_result.expected_tensor_keys)
            )

            self._pending_results.pop(model_pass_id)
            pending_result.future.set_result(result)

    @override
    async def notify_error(self, error: ModelPassError) -> None:
        model_pass_id = error.context.model_pass_id
        pending_result = self._pending_results.get(model_pass_id)

        if pending_result is None:
            raise ValueError(f"No pending result for model pass {model_pass_id}")

        if pending_result.future.done():
            return

        self._pending_results.pop(model_pass_id)
        pending_result.future.set_exception(
            RuntimeError(f"Model pass {model_pass_id} failed: {error.message}")
        )

    @override
    async def register_result_wait(
        self,
        request_id: ModelPassId,
        to_wait_list: list[TensorKey],
    ) -> asyncio.Future[TensorBundle]:
        if request_id in self._pending_results:
            raise ValueError(
                f"Result wait already registered for model pass {request_id}"
            )

        future: asyncio.Future[TensorBundle] = (
            asyncio.get_running_loop().create_future()
        )
        expected_tensor_keys = frozenset(to_wait_list)

        if not expected_tensor_keys:
            future.set_result(TensorBundle(bundle={}))
            return future

        self._pending_results[request_id] = DefaultResultGatherer._PendingResult(
            future=future,
            expected_tensor_keys=expected_tensor_keys,
            bundle=TensorBundle(bundle={}),
        )
        future.add_done_callback(
            lambda completed_future: self._cleanup_pending_result(
                request_id,
                completed_future,
            )
        )

        return future

    def _cleanup_pending_result(
        self,
        model_pass_id: ModelPassId,
        future: asyncio.Future[TensorBundle],
    ) -> None:
        pending_result = self._pending_results.get(model_pass_id)

        if pending_result is not None and pending_result.future is future:
            self._pending_results.pop(model_pass_id)
