from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import pytest

from distributed_inference.directory.application.ports.outbound.registry import (
    ServerRegistry,
    ServiceRegistry,
)
from distributed_inference.directory.application.ports.outbound.resolver import (
    ServiceResolver,
)
from distributed_inference.directory.domain.registration import (
    ServerRegistration,
    ServiceRegistration,
)
from distributed_inference.directory.domain.service_instance import (
    HostServiceEndpoint,
    ServiceInstance,
    ServiceProtocol,
    ServiceType,
    UriServiceEndpoint,
)
from distributed_inference.domain.identifiers import ServerId, ServiceId

type HostServiceProtocol = Literal[
    ServiceProtocol.HTTP,
    ServiceProtocol.GRPC,
    ServiceProtocol.ARROW,
]


@dataclass(frozen=True)
class DirectoryDependencies:
    server_registry: ServerRegistry
    service_registry: ServiceRegistry
    service_resolver: ServiceResolver


def build_server_id() -> ServerId:
    return ServerId(server_id=uuid4())


def build_service_instance(
    server_id: ServerId,
    *,
    service_type: ServiceType = ServiceType.MODEL_MANAGER,
    port: int = 8000,
    protocol: HostServiceProtocol = ServiceProtocol.HTTP,
) -> ServiceInstance:
    return ServiceInstance(
        service_id=ServiceId(
            server_id=server_id,
            service_id=uuid4(),
        ),
        service_type=service_type,
        service_endpoint=HostServiceEndpoint(
            host="model-manager",
            port=port,
            protocol=protocol,
        ),
    )


def build_uri_service_instance(
    server_id: ServerId,
    *,
    service_type: ServiceType = ServiceType.INFERENCE_SERVICE,
    port: int = 9090,
) -> ServiceInstance:
    return ServiceInstance(
        service_id=ServiceId(
            server_id=server_id,
            service_id=uuid4(),
        ),
        service_type=service_type,
        service_endpoint=UriServiceEndpoint(
            host="model-manager",
            port=port,
            protocol=ServiceProtocol.PYRO,
            object_identifier="distributed-inference.service",
        ),
    )


class DirectoryServiceContract:
    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_server_can_be_registered(
        self,
        directory: DirectoryDependencies,
    ) -> None:
        registration = ServerRegistration(server_id=build_server_id())

        await directory.server_registry.register_server(registration)

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_registered_service_resolves_by_id(
        self,
        directory: DirectoryDependencies,
    ) -> None:
        server_id = build_server_id()
        service = build_service_instance(server_id)

        await directory.server_registry.register_server(
            ServerRegistration(server_id=server_id)
        )
        await directory.service_registry.register_service(
            ServiceRegistration(service_instance=service)
        )

        assert (
            await directory.service_resolver.resolve_service_by_id(service.service_id)
            == service
        )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_resolve_by_server_is_scoped_to_that_server(
        self,
        directory: DirectoryDependencies,
    ) -> None:
        server_id = build_server_id()
        foreign_server_id = build_server_id()
        expected = {
            build_service_instance(server_id, port=8001),
            build_service_instance(
                server_id,
                service_type=ServiceType.INFERENCE_SERVICE,
                port=8002,
                protocol=ServiceProtocol.GRPC,
            ),
            build_uri_service_instance(server_id),
        }
        foreign_service = build_service_instance(foreign_server_id, port=9000)

        for registered_server_id in (server_id, foreign_server_id):
            await directory.server_registry.register_server(
                ServerRegistration(server_id=registered_server_id)
            )

        for service in (*expected, foreign_service):
            await directory.service_registry.register_service(
                ServiceRegistration(service_instance=service)
            )

        resolved = await directory.service_resolver.resolve_service_by_server(server_id)

        assert set(resolved) == expected

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_resolve_by_server_and_type_filters_services(
        self,
        directory: DirectoryDependencies,
    ) -> None:
        server_id = build_server_id()
        model_manager = build_service_instance(
            server_id,
            service_type=ServiceType.MODEL_MANAGER,
            port=8001,
        )
        inference_service = build_service_instance(
            server_id,
            service_type=ServiceType.INFERENCE_SERVICE,
            port=8002,
            protocol=ServiceProtocol.GRPC,
        )

        await directory.server_registry.register_server(
            ServerRegistration(server_id=server_id)
        )
        for service in (model_manager, inference_service):
            await directory.service_registry.register_service(
                ServiceRegistration(service_instance=service)
            )

        resolved = await directory.service_resolver.resolve_service_by_server_and_type(
            server_id,
            ServiceType.INFERENCE_SERVICE,
        )

        assert resolved == [inference_service]

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_missing_service_and_server_have_clear_results(
        self,
        directory: DirectoryDependencies,
    ) -> None:
        missing_server_id = build_server_id()
        missing_service_id = ServiceId(
            server_id=missing_server_id,
            service_id=uuid4(),
        )

        with pytest.raises(KeyError, match="not found"):
            await directory.service_resolver.resolve_service_by_id(missing_service_id)

        assert (
            await directory.service_resolver.resolve_service_by_server(
                missing_server_id
            )
            == []
        )
