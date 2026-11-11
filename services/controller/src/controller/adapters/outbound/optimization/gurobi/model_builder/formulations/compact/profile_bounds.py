from controller.adapters.outbound.optimization.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from controller.application.optimization.formulation.compact.profile_data import (
    ProfileDataError,
    fixed_shape_profile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)


def cut_upper_mb(profile: ModelVariantStaticProfile) -> float:
    maximum = fixed_shape_profile(profile).max_cut_size_mib
    if maximum is None:
        raise ProfileDataError("compact requires profiled max_cut_size_mib")
    return bytes_to_megabytes(maximum * 1024**2)
