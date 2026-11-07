from typing import override

from catalog.api.hop.hop_deployment_directory import HopDeploymentDirectory
from worker.application.failure.abc.failure_propagator import (
    FailurePropagator,
)
from worker.application.ports.outbound.failure_snd.failure_sender import FailureSender
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    FailurePlanReader,
)
from worker.domain.model.model_pass import ModelPassError


class DefaultFailurePropagator(FailurePropagator):
    def __init__(
        self,
        failure_plan_reader: FailurePlanReader,
        hop_directory: HopDeploymentDirectory,
        failure_sender: FailureSender,
    ) -> None:
        super().__init__()
        self._failure_plan_reader = failure_plan_reader
        self._hop_directory = hop_directory
        self._failure_sender = failure_sender

    @override
    async def propagate_failure(self, error: ModelPassError) -> None:

        plan_version = error.plan_version
        failure_plan = await self._failure_plan_reader.get_failure_plan_by_version(
            plan_version
        )
        flow_id = error.flow_id
        sink_replica_id = failure_plan.get_sink_replica_id_for_flow(flow_id)
        worker_id = await self._hop_directory.get_worker_for_planned_invocable_hop(
            plan_version, sink_replica_id
        )

        await self._failure_sender.send_failure(error, worker_id)
