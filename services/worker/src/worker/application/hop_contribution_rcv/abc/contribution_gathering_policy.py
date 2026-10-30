from abc import ABC, abstractmethod
from dataclasses import dataclass

from worker.domain.context.model_pass_context import ModelPassContext
from worker.domain.hop.hop import DestinationHop
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


@dataclass(frozen=True, slots=True)
class HopContributionCollectionKey:
    model_pass_context: ModelPassContext
    hop: DestinationHop


@dataclass(frozen=True, slots=True)
class HopContributionCollection:
    key: HopContributionCollectionKey
    collection: tuple[HopInvocationContribution, ...]


class ContributionGatheringPolicy(ABC):
    @abstractmethod
    async def key_of(
        self, contribution: HopInvocationContribution
    ) -> HopContributionCollectionKey: ...

    @abstractmethod
    async def gather(
        self, contributions: list[HopInvocationContribution]
    ) -> HopContributionCollection: ...

    @abstractmethod
    async def is_ready(
        self, contributions: list[HopInvocationContribution]
    ) -> bool: ...

    @abstractmethod
    async def validate(
        self,
        other_contributions: list[HopInvocationContribution],
        new_contribution: HopInvocationContribution,
    ) -> None: ...
