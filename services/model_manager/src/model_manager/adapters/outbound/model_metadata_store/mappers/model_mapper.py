from integration.dto.model.model_dto import ModelDto
from model_manager.domain.model import Model, ModelInfo


class ModelMapper:
    @staticmethod
    def to_model_dto(model: Model) -> ModelDto:
        model_id = model.model_id
        model_task = model.model_info.model_task
        architecture_info = model.model_info.architecture_info

        return ModelDto(
            model_id=model_id,
            model_task=model_task,
            architecture_info=architecture_info,
        )

    @staticmethod
    def from_model_dto(dto: ModelDto) -> Model:
        return Model(
            model_id=dto.model_id,
            model_info=ModelInfo(
                model_task=dto.model_task,
                architecture_info=dto.architecture_info,
            ),
        )
