# mypy: disable-error-code="attr-defined,name-defined,misc"

from pyarrow import flight

from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_contribution_sink import (
    ArrowModelPassResultContributionSink,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retriever import (
    ArrowModelPassResultRetriever,
)


class ArrowModelPassResultServer(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        contribution_sink: ArrowModelPassResultContributionSink,
        result_retriever: ArrowModelPassResultRetriever,
    ) -> None:
        self._contribution_sink = contribution_sink
        self._result_retriever = result_retriever
        super().__init__(location)

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        self._contribution_sink.do_put(context, descriptor, reader, writer)

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightStreamWriter,
    ) -> None:
        self._result_retriever.do_exchange(context, descriptor, reader, writer)
