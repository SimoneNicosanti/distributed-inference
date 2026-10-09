from typing import Any, override

from ingress.application.abc.result_collector import ResultCollector
from ingress.application.ports.inbound.result_receiver import ResultReceiver


class DefaultResultReceiver(ResultReceiver):
    def __init__(self, result_collector: ResultCollector) -> None:
        super().__init__()
        self._gatherer = result_collector

    @override
    async def receive_result(self, request_id: Any, result_data: Any) -> None:
        await self._gatherer.add_result_contribution(request_id, result_data)

        return None
