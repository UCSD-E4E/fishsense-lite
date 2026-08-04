"""Fish model for the FishSense API."""

from sqlmodel import Field, SQLModel, UniqueConstraint


class Fish(SQLModel, table=True):
    """Fish model representing a fish in the database."""

    # `name` is the natural key for a *physical fish model* (Grouper, Purple
    # Angel, …): the same model photographed across dives/cameras/time resolves
    # to ONE Fish, so its measurements accumulate under one identity. Real
    # (wild) fish leave `name` NULL and keep their per-cluster identity — and
    # Postgres/SQLite both allow multiple NULLs under a UNIQUE constraint, so
    # unnamed real fish never collide. `species_id` is NULL for models (there is
    # no Species row for a model).
    __table_args__ = (UniqueConstraint("name", name="uq_fish_name"),)

    id: int | None = Field(default=None, primary_key=True)

    name: str | None = Field(default=None)

    species_id: int | None = Field(default=None, foreign_key="species.id")
