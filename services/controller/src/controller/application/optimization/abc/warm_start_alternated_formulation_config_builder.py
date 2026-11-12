from abc import abstractmethod
from collections.abc import Generator


class WarmStartFormulationConfigBuilder[FormulationConfigT]:
    @abstractmethod
    def warm_start_formulation_configs(
        self, full_config: FormulationConfigT
    ) -> Generator[FormulationConfigT]: ...
