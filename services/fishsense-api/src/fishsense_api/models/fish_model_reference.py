"""Known-size reference for the physical fish models."""

from sqlmodel import Field, SQLModel, UniqueConstraint


class FishModelReference(SQLModel, table=True):
    """Ground-truth length of a physical fish model, keyed by the same `name`
    natural key stage 14 resolves model identity by (`Fish.name`).

    These lengths are the pipeline's held-out **validation** set: they are
    compared against what stage 14 measures and are NEVER fed into laser
    calibration — doing so would make the benchmark grade itself. The
    `fish_model_measurement_accuracy` view joins measurements to these rows.
    """

    __table_args__ = (UniqueConstraint("name", name="uq_fish_model_reference_name"),)

    id: int | None = Field(default=None, primary_key=True)

    # Matches `Fish.name` / the `Fish Model, <name>` species-label leaf.
    name: str = Field(index=True)

    known_length_m: float

    # Provenance: how/when the length was established (e.g. field dates,
    # caliper session). Free text — the measurement itself is the datum.
    notes: str | None = Field(default=None)
