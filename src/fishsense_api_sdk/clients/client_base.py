"""Base client for interacting with the Fishsense API."""

import asyncio
import base64
from abc import ABC
from logging import Logger, getLogger

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

    @property
    def logger(self) -> Logger:
        """Logger for the client."""
        return self.__logger

    def __init__(
        self,
        base_url: str,
        username: str | None,
        password: str | None,
        timeout: int,
        semaphore: asyncio.Semaphore,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.base_url = base_url
        self.timeout = timeout
        self.semaphore = semaphore

        self.__token = (
            base64.b64encode(f"{username}:{password}".encode("utf-8"))
            if username and password
            else None
        )
        self.__client_internal: httpx.AsyncClient | None = None
        self.__inside_context = False
        self.__logger = getLogger(__name__)

    def __create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __aenter__(self) -> "ClientBase":
        self.logger.debug("Entering async context manager for ClientBase")
        self.__inside_context = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.__client_internal is not None:
            self.logger.debug("Exiting async context manager for ClientBase")
            self.__inside_context = False
            await self.__client_internal.aclose()
            self.__client_internal = None

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _get(self, endpoint: str) -> httpx.Response:
        async with self.semaphore:
            self.logger.debug("GET request to %s", endpoint)
            return await self.__client.get(
                endpoint,
                headers=(
                    {"Authorization": f"Basic {self.__token.decode('utf-8')}"}
                    if self.__token
                    else {}
                ),
            )

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _post(self, endpoint: str, json: dict) -> httpx.Response:
        async with self.semaphore:
            self.logger.debug("POST request to %s with payload: %s", endpoint, json)
            return await self.__client.post(
                endpoint,
                json=json,
                headers=(
                    {"Authorization": f"Basic {self.__token.decode('utf-8')}"}
                    if self.__token
                    else {}
                ),
            )

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def _put(self, endpoint: str, json: dict) -> httpx.Response:
        async with self.semaphore:
            self.logger.debug("PUT request to %s with payload: %s", endpoint, json)
            return await self.__client.put(
                endpoint,
                json=json,
                headers=(
                    {"Authorization": f"Basic {self.__token.decode('utf-8')}"}
                    if self.__token
                    else {}
                ),
            )
