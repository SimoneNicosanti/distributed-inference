from abc import ABC, abstractmethod

from utils.raises.raises import raises
from worker.domain.activity.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityRequest,
)
from worker.domain.activity.error.activity_grant_acquisition_error import (
    ActivityGrantAcquisitionError,
)


class ActivityManager(ABC):
    @abstractmethod
    @raises(ActivityGrantAcquisitionError)
    async def acquire_activity_grant(
        self,
        request: ActivityRequest,
    ) -> ActivityGrant:
        """
        ## Raises:
            ActivityGrantAcquisitionError
        """
        ...

    @abstractmethod
    async def release_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> None: ...
