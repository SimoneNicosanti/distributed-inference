import contextlib
from typing import override

from worker.application.ports.inbound.failure_rcv.failure_reciever import (
    FailureReceiver,
)
from worker.application.postbox.abc.outbox import InvalidOutboxDepositError, Outbox
from worker.domain.model.model_pass import ModelPassError, ModelPassOutcome


class DefaultFailureReceiver(FailureReceiver):
    def __init__(self, outbox: Outbox[ModelPassOutcome]) -> None:
        super().__init__()
        self._outbox = outbox

    @override
    async def notify_failure(self, error: ModelPassError) -> None:
        with contextlib.suppress(InvalidOutboxDepositError):
            await self._outbox.deposit(error)
