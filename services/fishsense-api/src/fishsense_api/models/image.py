"""Model representing an image."""

from datetime import datetime

from sqlalchemy import Index, text
from sqlmodel import DateTime, Field

from fishsense_api.models.model_base import ModelBase


class Image(ModelBase, table=True):
    """Model representing an image."""

    # At most one CANONICAL row per checksum.
    #
    # The same physical frames legitimately appear under several dive rows
    # (prod dives 64 and 66 are both `082929_FishModels_FSL07`), so `checksum`
    # itself is deliberately non-unique. What must be unique is which of those
    # copies is the canonical one.
    #
    # `post_image` computes that with a read-then-write ("is there already a
    # row with this checksum?"), which under READ COMMITTED does not stop two
    # concurrent requests from both concluding they are first. Only the
    # database can. A losing racer gets an IntegrityError, and because the
    # ingest activity retries, the retry re-reads, sees the winner, and
    # correctly writes `is_canonical=False` -- so the race self-heals rather
    # than needing a resolution path in application code.
    #
    # Partial index: rows with `is_canonical=False` are unconstrained, which is
    # exactly the duplicate case. Supported by both Postgres and SQLite.
    __table_args__ = (
        Index(
            "uq_image_canonical_checksum",
            "checksum",
            unique=True,
            postgresql_where=text("is_canonical"),
            sqlite_where=text("is_canonical"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    path: str = Field(max_length=255, unique=True, index=True)
    taken_datetime: datetime = Field(sa_type=DateTime(timezone=True), default=None)
    checksum: str = Field(max_length=32, index=True)
    is_canonical: bool = Field(default=False)

    dive_id: int | None = Field(default=None, foreign_key="dive.id")
    camera_id: int | None = Field(default=None, foreign_key="camera.id")
