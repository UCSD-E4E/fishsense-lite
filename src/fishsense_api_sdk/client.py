""" "Main client for interacting with the Fishsense API."""

import asyncio

from fishsense_api_sdk.clients.camera_client import CameraClient
from fishsense_api_sdk.clients.dive_client import DiveClient
from fishsense_api_sdk.clients.image_client import ImageClient
from fishsense_api_sdk.clients.label_client import LabelClient
from fishsense_api_sdk.clients.user_client import UserClient


class Client:
    """Main client for interacting with the Fishsense API."""

    @property
    def cameras(self) -> CameraClient:
        """Get a list of camera instances .

        Returns:
            CameraClient: The camera client instance.
        """
        return self.__cameras

    @property
    def dives(self) -> DiveClient:
        """Get a list of dive instances .

        Returns:
            DiveClient: The dive client instance.
        """
        return self.__dives

    @property
    def images(self) -> ImageClient:
        """Get a list of image instances .

        Returns:
            ImageClient: The image client instance.
        """
        return self.__images

    @property
    def labels(self) -> LabelClient:
        """Get a list of label instances .

        Returns:
            LabelClient: The label client instance.
        """
        return self.__labels

    @property
    def users(self) -> UserClient:
        """Get a list of user instances .

        Returns:
            UserClient: The user client instance.
        """
        return self.__users

    def __init__(
        self, base_url: str, timeout: int = 10, max_concurrent_requests: int = 10
    ):
        self.base_url = base_url

        self.__semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.__cameras = CameraClient(base_url, timeout, self.__semaphore)
        self.__dives = DiveClient(base_url, timeout, self.__semaphore)
        self.__images = ImageClient(base_url, timeout, self.__semaphore)
        self.__labels = LabelClient(base_url, timeout, self.__semaphore)
        self.__users = UserClient(base_url, timeout, self.__semaphore)

    async def __aenter__(self):
        await self.cameras.__aenter__()
        await self.dives.__aenter__()
        await self.images.__aenter__()
        await self.labels.__aenter__()
        await self.users.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.cameras.__aexit__(exc_type, exc_value, traceback)
        await self.dives.__aexit__(exc_type, exc_value, traceback)
        await self.images.__aexit__(exc_type, exc_value, traceback)
        await self.labels.__aexit__(exc_type, exc_value, traceback)
        await self.users.__aexit__(exc_type, exc_value, traceback)
