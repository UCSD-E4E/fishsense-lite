"""Base client for interacting with the Fishsense API."""

from abc import ABC

import httpx


class ClientBase(ABC):
    # pylint: disable=too-few-public-methods
    """Base client for interacting with the Fishsense API."""

    @property
    def _client(self) -> httpx.AsyncClient:
        return self.__client

    def __init__(self, base_url: str, timeout):
        self.base_url = base_url
        self.timeout = timeout

        self.__client: httpx.AsyncClient | None = None

    def __create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __aenter__(self) -> "ClientBase":
        self.__client = self.__create_client()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.__client is not None:
            await self.__client.aclose()
            self.__client = None
