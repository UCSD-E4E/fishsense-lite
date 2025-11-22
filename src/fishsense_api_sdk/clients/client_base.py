"""Base client for interacting with the Fishsense API."""

import asyncio
from abc import ABC

import httpx
from retry import retry


class ClientBase(ABC):
    # pylint: disable=too-few-public-methods
    """Base client for interacting with the Fishsense API."""

    @property
    def __client(self) -> httpx.AsyncClient:
        if not self.__inside_context:
            raise RuntimeError(
                "Client must be used within an async context manager. "
                "Use 'async with' to create a context."
            )

        if self.__client_internal is None:
            self.__client_internal = self.__create_client()

        return self.__client_internal

    def __init__(self, base_url: str, timeout: int, semaphore: asyncio.Semaphore):
        self.base_url = base_url
        self.timeout = timeout
        self.semaphore = semaphore

        self.__client_internal: httpx.AsyncClient | None = None
        self.__inside_context = False

    def __create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __aenter__(self) -> "ClientBase":
        self.__inside_context = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.__client_internal is not None:
            await self.__client_internal.aclose()
            self.__client_internal = None

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _get(self, endpoint: str) -> httpx.Response:
        async with self.semaphore:
            return await self.__client.get(endpoint)

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _post(self, endpoint: str, json: dict) -> httpx.Response:
        async with self.semaphore:
            return await self.__client.post(endpoint, json=json)

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _put(self, endpoint: str, json: dict) -> httpx.Response:
        async with self.semaphore:
            return await self.__client.put(endpoint, json=json)
