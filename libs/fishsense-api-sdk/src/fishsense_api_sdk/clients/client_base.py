"""Base client for interacting with the Fishsense API."""

import asyncio
import base64
from abc import ABC
from logging import Logger, getLogger

import httpx

# Retry tuning. Module-level so tests can zero the delay without sleeping.
_MAX_ATTEMPTS = 3
_INITIAL_DELAY_S = 2.0
_BACKOFF = 2.0

# Server-side failures worth replaying. 5xx only — a 4xx is an answer, and
# retrying it just adds latency before the same result.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

# Transport failures worth replaying for an idempotent request: the response
# never arrived, so the outcome is unknown but replaying is harmless.
_RETRYABLE_TRANSPORT = (httpx.TransportError,)

# For a non-idempotent request only a `ConnectError` is safe: it proves the
# connection was never established, so the server cannot have applied the
# write. A `ReadTimeout` proves nothing — the request may well have been
# committed before the response was lost.
_RETRYABLE_TRANSPORT_UNSAFE_METHOD = (httpx.ConnectError,)


class ClientBase(ABC):
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Base client for interacting with the Fishsense API.

    **Retries.** `_get` / `_put` / `_delete` replay transient failures — 5xx
    responses and transport errors — up to `_MAX_ATTEMPTS` with exponential
    backoff. `_post` does not: `post_species` / `post_fish` / `post_cluster`
    create rows, and a 5xx can be returned by a server that already committed,
    so a blind replay would duplicate the write. POST retries only on
    `ConnectError`, where nothing can have reached the server.

    Exhausting retries on a status returns the failing response rather than
    raising — callers inspect `status_code` themselves (`dive_client.get`
    treats 404 as "no such dive") and call `raise_for_status()` when they want
    the exception. A transport error that never yields a response is re-raised.

    This replaces four `@retry(exceptions=httpx.HTTPStatusError, tries=3)`
    decorators that never fired. `retry` is synchronous, so wrapping an
    `async def` returned the coroutine before it could raise — measured at one
    attempt, not three — and `HTTPStatusError` is raised only by
    `raise_for_status()`, which lives in the subclasses, never in the
    decorated methods. The dead decorators mattered because
    `_retry_policies.SDK_FAIL_FAST_RETRY_POLICY` marks `HTTPStatusError`
    non-retryable on the strength of them, so a single transient 5xx failed a
    whole activity with no retry at any layer.
    """

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
        transport: httpx.AsyncBaseTransport | None = None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.base_url = base_url
        self.timeout = timeout
        self.semaphore = semaphore
        # Optional httpx transport. Production leaves this None and gets the
        # default network transport; passing `httpx.ASGITransport(app=...)`
        # drives a FastAPI app in-process, which is how the SDK<->API contract
        # is tested without standing up a server. Mocking `_post` cannot catch
        # a URL or payload the API would actually reject.
        self.transport = transport

        self.__token = (
            base64.b64encode(f"{username}:{password}".encode("utf-8"))
            if username and password
            else None
        )
        self.__client_internal: httpx.AsyncClient | None = None
        self.__inside_context = False
        self.__logger = getLogger(__name__)

    def __create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        )

    def __headers(self) -> dict:
        if self.__token is None:
            return {}
        return {"Authorization": f"Basic {self.__token.decode('utf-8')}"}

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json: dict | None = None,
        idempotent: bool,
    ) -> httpx.Response:
        """Issue one request, replaying transient failures.

        The semaphore is held only for the call itself — backoff sleeps
        happen outside it, so a retrying request doesn't occupy a concurrency
        slot while it waits.
        """
        retryable_exc = (
            _RETRYABLE_TRANSPORT if idempotent else _RETRYABLE_TRANSPORT_UNSAFE_METHOD
        )
        delay = _INITIAL_DELAY_S

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            last_attempt = attempt == _MAX_ATTEMPTS
            try:
                async with self.semaphore:
                    self.logger.debug("%s %s (attempt %d)", method, endpoint, attempt)
                    response = await self.__client.request(
                        method, endpoint, json=json, headers=self.__headers()
                    )
            except retryable_exc as exc:
                if last_attempt:
                    raise
                self.logger.warning(
                    "%s %s failed (%s); retrying in %.1fs",
                    method,
                    endpoint,
                    exc.__class__.__name__,
                    delay,
                )
            else:
                retryable_status = idempotent and response.status_code in (
                    _RETRYABLE_STATUS
                )
                if not retryable_status or last_attempt:
                    return response
                self.logger.warning(
                    "%s %s returned %d; retrying in %.1fs",
                    method,
                    endpoint,
                    response.status_code,
                    delay,
                )

            await asyncio.sleep(delay)
            delay *= _BACKOFF

        # Unreachable: the loop either returns or raises on the last attempt.
        raise AssertionError("retry loop exited without a result")  # pragma: no cover

    async def _get(self, endpoint: str) -> httpx.Response:
        return await self._request("GET", endpoint, idempotent=True)

    async def _post(self, endpoint: str, json: dict) -> httpx.Response:
        return await self._request("POST", endpoint, json=json, idempotent=False)

    async def _put(self, endpoint: str, json: dict | None = None) -> httpx.Response:
        return await self._request("PUT", endpoint, json=json, idempotent=True)

    async def _delete(self, endpoint: str) -> httpx.Response:
        return await self._request("DELETE", endpoint, idempotent=True)

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
