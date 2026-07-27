from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import ServerId


class ServerRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: ServerId
