"""Fixed-B6 experiment settings; candidate-count sweeps are unavailable."""

from typing import Literal, Self

from pydantic import model_validator

from scalability_benchmark.configuration import (
    CampaignSettings,
    ExperimentConfig,
    PilotCase,
    PilotSettings,
)


class ModelVariantCampaignSettings(CampaignSettings):
    models: tuple[Literal[1], ...] = (1,)


class ModelVariantPilotCase(PilotCase):
    models: Literal[1] = 1


class ModelVariantPilotSettings(PilotSettings):
    cases: tuple[ModelVariantPilotCase, ...]


class ModelVariantExperimentConfig(ExperimentConfig):
    model_selection: Literal["model_variant"] = "model_variant"
    main: ModelVariantCampaignSettings
    sensitivity: ModelVariantCampaignSettings
    pilot: ModelVariantPilotSettings

    @model_validator(mode="after")
    def fixed_catalog(self) -> Self:
        if tuple(model.name for model in self.catalog.models) != ("efficientnet_b6",):
            raise ValueError(
                "The model_variant experiment requires only efficientnet_b6 in its catalog"
            )
        return self
