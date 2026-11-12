from collections.abc import Generator
from typing import override

from controller.application.optimization.strategies.alternated.abc.phase_generator import (
    PhaseGenerator,
    PhaseInfo,
)


class CompactPhaseGenerator(PhaseGenerator):
    @override
    def next_phase(self) -> Generator[PhaseInfo]:

        return super().next_phase()
