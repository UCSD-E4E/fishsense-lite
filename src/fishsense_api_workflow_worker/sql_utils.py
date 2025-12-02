# """SQL Utilities"""

# import logging
# import httpx
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Union

# import psycopg

# from fishsense_api_workflow_worker.config import IS_DOCKER

# # from fishsense_data_processing_spider.metrics import get_summary

# __log = logging.getLogger("sql_utils")


# def load_query(path: Path) -> str:
#     """Loads query from path

#     Args:
#         path (Path): Path to query file

#     Returns:
#         str: Query contents
#     """

#     if IS_DOCKER and not path.is_absolute():
#         # If running in Docker, we need to resolve the path relative to the
#         # container's working directory
#         path = Path("/app") / path

#     with open(path, "r", encoding="utf-8") as handle:
#         return handle.read(int(1e9))


# def do_query(
#     path: Union[Path, str], cur: psycopg.Cursor, params: Optional[Dict[str, Any]] = None
# ):
#     """Convenience function to time and execute a query

#     Args:
#         path (Union[Path, str]): Path to query file
#         cur (psycopg.Cursor): Cursor
#         params (Optional[Dict[str, Any]]): Query parameters.  Defaults to None
#     """
#     path = Path(path)
#     # query_timer = get_summary(
#     #     'query_duration'
#     # )
#     # with query_timer.labels(query=path.stem).time():
#     try:
#         cur.execute(query=load_query(path), params=params)
#     except psycopg.errors.Error as exc:
#         __log.exception("Query %s with params %s failed due to %s", path, params, exc)
#         raise exc


# def do_many_query(
#     path: Union[Path, str],
#     cur: psycopg.Cursor,
#     param_seq: List[Dict[str, Any]],
#     returning: bool = False,
# ) -> None:
#     """Convenience function to time and executemany

#     Args:
#         path (Union[Path, str]): Path to query file
#         cur (psycopg.Cursor): Cursor
#         param_seq (List[Dict[str, Any]]): Query parameters
#         returning (bool, optional): Flag indicating whether or not this query returns data. Defaults
#         to False.
#     """
#     path = Path(path)
#     # query_timer = get_summary(
#     #     'query_duration'
#     # )
#     # with query_timer.labels(query=path.stem).time():
#     try:
#         cur.executemany(
#             query=load_query(path), params_seq=param_seq, returning=returning
#         )
#     except psycopg.errors.Error as exc:
#         __log.exception(
#             "Query %s with param seq %s failed due to %s", path, param_seq, exc
#         )
#         raise exc


"""SQL Utilities"""

import logging
import httpx
from typing import Any, Dict, List, Optional

__log = logging.getLogger("sql_utils")

API_BASE_URL = "http://fishsense-api:8000/api/v1"  # Internal Docker network URL

async def do_query(endpoint: str, params: Optional[Dict[str, Any]] = None):
    """Perform a query via the fishsense-api.

    Args:
        endpoint (str): The API endpoint to call (e.g., "cameras").
        params (Optional[Dict[str, Any]]): Query parameters. Defaults to None.

    Returns:
        dict: The API response.
    """
    if not endpoint.endswith("/"):
        endpoint += "/"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            __log.exception("API call to %s failed due to %s", endpoint, exc)
            raise exc


async def do_many_query(endpoint: str, data: List[Dict[str, Any]]):
    """Perform a batch query via the fishsense-api.

    Args:
        endpoint (str): The API endpoint to call (e.g., "cameras").
        data (List[Dict[str, Any]]): The data to send in the request.

    Returns:
        dict: The API response.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            __log.exception("Batch API call to %s failed due to %s", endpoint, exc)
            raise exc