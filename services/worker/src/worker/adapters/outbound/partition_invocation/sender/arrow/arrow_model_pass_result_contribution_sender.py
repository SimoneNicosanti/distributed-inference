# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import ModelPassResultContributionMetadataDto
from integration.service_instance.ingress_instance import ResultSinkInstance
from integration.service_instance.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    SUBMIT_CONTRIBUTION_COMMAND,
    build_result_payload_schema,
)
from worker.adapters.context_mapper import model_pass_context_to_dto
from worker.application.ports.outbound.catalog.service.service_directory_gateway import (
    ServiceDirectoryGateway,
)
from worker.application.ports.outbound.partition_invocation.sender.contribution_sender import (
    ContributionSender,
)
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class ArrowModelPassResultContributionSender(
    ContributionSender[ModelPassResultContribution]
):
    def __init__(
        self,
        directory: ServiceDirectoryGateway,
        timeout_s: float = 30.0,
    ) -> None:
        self._directory = directory
        self._timeout_s = timeout_s

    @override
    async def send(self, contribution: ModelPassResultContribution) -> None:
        result_sink_id = (
            contribution.model_pass_context.model_invocation_context.result_sink
        )
        result_sink_instance = await self._directory.get_result_sink_instance_by_id(
            result_sink_id
        )

        interface = self._select_arrow_interface(result_sink_instance)

        await asyncio.to_thread(
            self._send_sync,
            interface,
            contribution,
        )

    @staticmethod
    def _select_arrow_interface(
        result_sink_instance: ResultSinkInstance,
    ) -> CapabilityInterface:
        interfaces = (
            result_sink_instance.get_capability_interfaces_by_type_and_protocol(
                CapabilityType.RESULT_SINKER,
                CapabilityProtocol.ARROW,
            )
        )

        if not interfaces:
            raise ValueError(
                f"Result sink {result_sink_instance.service_id} "
                "does not expose an Arrow result interface"
            )

        return interfaces[0]

    def _send_sync(
        self,
        interface: CapabilityInterface,
        contribution: ModelPassResultContribution,
    ) -> None:
        batch = self._encode_contribution(contribution)

        descriptor = flight.FlightDescriptor.for_command(SUBMIT_CONTRIBUTION_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)

        endpoint = interface.endpoint.unicode_string()

        with closing(flight.connect(endpoint)) as client:
            writer, _ = client.do_put(
                descriptor,
                batch.schema,
                options=options,
            )

            with closing(writer):
                writer.write_batch(batch)
                writer.done_writing()

    @staticmethod
    def _encode_contribution(
        contribution: ModelPassResultContribution,
    ) -> pa.RecordBatch:
        tensor_values = {
            name: np.ascontiguousarray(tensor.as_numpy())
            for name, tensor in contribution.bundle.bundle.items()
        }

        schema = build_result_payload_schema(tensor_values)
        metadata = ModelPassResultContributionMetadataDto(
            model_pass_context=model_pass_context_to_dto(
                contribution.model_pass_context
            )
        )

        arrays: list[pa.Array] = [
            pa.array(
                [metadata.model_dump_json()],
                type=pa.large_string(),
            )
        ]

        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        )

        return pa.RecordBatch.from_arrays(
            arrays,
            schema=schema,
        )
