from typing import Any, override

from ingress.application.invocation_manager import InvocationManager
from ingress.application.ports.outbound.invoker import Invoker
from ingress.application.result_collector import ResultCollector


class DefaultInvocationManager(InvocationManager):
    def __init__(self, result_collector: ResultCollector, invoker: Invoker) -> None:

        super().__init__()
        self._invoker = invoker
        self._result_collector = result_collector

    @override
    async def invoke(self, request_data: Any) -> Any:

        ## Build request from request data
        request: Any = None
        request_id: Any = None

        ## Get to_wait_list from model info
        to_wait_list: list[Any] = []

        ## Use the invoker port to do the actual invocation
        await self._invoker.invoke(request)

        ## Await on the collector to get the results
        result_data = await self._result_collector.collect_result(
            request_id, to_wait_list
        )

        return result_data
