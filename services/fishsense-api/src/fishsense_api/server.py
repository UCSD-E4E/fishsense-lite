"""FishSense API Server"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from fishsense_api.__version__ import __version__
from fishsense_api.database import DATABASE


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Generate the database lifespan.

    Args:
        app (FastAPI): FastAPI application instance.
    """
    database = DATABASE

    # Startup events (e.g., create tables)
    async with database.engine.begin() as conn:
        await database.init_database(conn)

    yield

    # Shutdown events (e.g., dispose engine)
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
