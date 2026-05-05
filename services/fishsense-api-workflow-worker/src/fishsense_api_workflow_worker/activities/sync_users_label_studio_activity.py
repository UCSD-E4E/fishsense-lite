"""Activity to sync users from Label Studio to Fishsense API."""

import asyncio

from fishsense_api_sdk.models.user import User
from httpx import HTTPStatusError
from label_studio_sdk import LseUserApi
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client, get_ls_client
from fishsense_api_workflow_worker.config import settings
from fishsense_shared import ExceptionGroupErrorLogging


def __from_label_studio(user: LseUserApi) -> User:
    """Create a User instance from a Label Studio user."""

    return User(
        id=None,
        label_studio_id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        last_activity=user.last_activity,
        date_joined=user.date_joined,
    )


@activity.defn
async def sync_users_label_studio_activity():
    """Activity to sync users from Label Studio to Fishsense API."""
    ls = get_ls_client()
    label_studio_users = await asyncio.to_thread(ls.users.list)

    activity.logger.info(
        f"Fetched {len(label_studio_users)} users from Label Studio at {settings.label_studio.url}"
    )

    async with get_fs_client() as fs:
        async with ExceptionGroupErrorLogging(activity.logger):
            async with asyncio.TaskGroup() as tg:
                for label_studio_user in label_studio_users:
                    if activity.is_cancelled():
                        activity.logger.info(
                            "Activity cancelled, stopping user sync from Label Studio"
                        )
                        return

                    try:
                        fs_user = await fs.users.get_by_email(label_studio_user.email)
                    except HTTPStatusError as e:
                        if e.response.status_code == 404:
                            fs_user = None
                        else:
                            raise

                    if fs_user is None:
                        tg.create_task(
                            fs.users.post(__from_label_studio(label_studio_user))
                        )
                    else:
                        new_fs_user = __from_label_studio(label_studio_user)
                        new_fs_user.id = fs_user.id

                        tg.create_task(fs.users.put(new_fs_user))

                    activity.heartbeat()
