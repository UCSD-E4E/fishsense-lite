"""FishSense API Server"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from fishsense_api.__version__ import __version__
from fishsense_api.database import run_alembic_upgrade, setup_database

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Generate the database lifespan.

    Order is intentional:
      1. `init_database` (`SQLModel.metadata.create_all`) — bootstraps
         tables on a fresh environment. The alembic baseline migration
         only contains `alter_column` ops, so it assumes tables already
         exist; running alembic against an empty DB would fail without
         this prior step. No-op against an existing prod schema.
      2. `run_alembic_upgrade` — applies any pending migrations
         (including non-table artifacts like the `dive_pipeline_status`
         view that aren't part of `SQLModel.metadata`). Sync API,
         offloaded via `asyncio.to_thread` so DDL doesn't block the
         event loop.
    """
    _log.info("fishsense-api starting (version=%s)", __version__)
    database = setup_database()

    _log.info("running SQLModel.metadata.create_all")
    async with database.engine.begin() as conn:
        await database.init_database(conn)

    _log.info("running alembic upgrade")
    await asyncio.to_thread(run_alembic_upgrade)
    _log.info("fishsense-api startup complete")

    yield

    _log.info("fishsense-api shutting down")
    await database.dispose()


app = FastAPI(lifespan=lifespan, version=__version__)


@app.get("/")
def home():
    """Home endpoint providing basic API information."""

    return {
        "message": "Welcome to the FishSense API!",
        "docs": "/docs",
        "version": __version__,
    }
