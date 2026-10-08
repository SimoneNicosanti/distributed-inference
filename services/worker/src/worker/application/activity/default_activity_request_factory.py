from typing import override

from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.domain.activity.activity_request import (
    ActivityRequest,
    ActivityType,
    ResourceRequirement,
    ResourceType,
)


class DefaultActivityRequestFactory(ActivityRequestFactory):
    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def create_request_for_activity_type(
        activity_type: ActivityType,
    ) -> ActivityRequest:

        match activity_type:
            case ActivityType.EXECUTION_INFERENCE:
                return (
                    DefaultActivityRequestFactory._build_execution_inference_request()
                )
            case ActivityType.NETWORK_TRANSMISSION:
                return (
                    DefaultActivityRequestFactory._build_network_transmission_request()
                )
            case ActivityType.NETWORK_PROFILING:
                return DefaultActivityRequestFactory._build_network_profiling_request()
            case ActivityType.EXECUTION_PROFILING:
                return (
                    DefaultActivityRequestFactory._build_execution_profiling_request()
                )
            case _:
                raise ValueError(f"Invalid activity type: {activity_type}")

    ## TODO: Check coherence with the activity manager
    @staticmethod
    def _build_network_transmission_request() -> ActivityRequest:
        required_lock = {
            ResourceType.NETWORK: ResourceRequirement(
                quantity=0.1,  ## TODO: Check this value: put a small value here, but we should check the sematic better
                exclusive=False,
            )
        }
        activity_request = ActivityRequest(
            activity_type=ActivityType.NETWORK_TRANSMISSION,
            resource_requirements=required_lock,
        )

        return activity_request

    @staticmethod
    def _build_network_profiling_request() -> ActivityRequest:
        requirements = {
            ResourceType.NETWORK: ResourceRequirement(
                quantity=0,
                exclusive=True,
            )
        }
        activity_request = ActivityRequest(
            activity_type=ActivityType.NETWORK_PROFILING,
            resource_requirements=requirements,
        )

        return activity_request

    @staticmethod
    def _build_execution_profiling_request() -> ActivityRequest:
        requirements = {
            ResourceType.COMPUTE: ResourceRequirement(quantity=0, exclusive=True)
        }
        return ActivityRequest(
            activity_type=ActivityType.EXECUTION_PROFILING,
            resource_requirements=requirements,
        )

    @staticmethod
    def _build_execution_inference_request() -> ActivityRequest:
        requirements = {
            ResourceType.COMPUTE: ResourceRequirement(quantity=0, exclusive=True)
        }
        activity_request = ActivityRequest(
            activity_type=ActivityType.EXECUTION_INFERENCE,
            resource_requirements=requirements,
        )

        return activity_request
