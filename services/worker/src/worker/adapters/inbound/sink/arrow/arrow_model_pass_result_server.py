# mypy: disable-error-code="attr-defined,name-defined,misc"

from pyarrow import flight

from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retriever import (
    ArrowModelPassResultRetriever,
)


class ArrowModelPassResultServer(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        result_retriever: ArrowModelPassResultRetriever,
    ) -> None:
        self._result_retriever = result_retriever
        super().__init__(location)

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightStreamWriter,
    ) -> None:
        self._result_retriever.do_exchange(context, descriptor, reader, writer)
