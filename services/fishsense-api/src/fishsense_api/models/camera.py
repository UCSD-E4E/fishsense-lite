"""Model representing a camera."""

from sqlmodel import Field, SQLModel, UniqueConstraint


class Camera(SQLModel, table=True):
    """Model representing a camera."""

    __table_args__ = (
        UniqueConstraint(
            "serial_number",
            name="uq_camera_serial_number",
        ),
        UniqueConstraint(
            "name",
            name="uq_camera_name",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    serial_number: str = Field(unique=True, index=True)
    name: str = Field(unique=True, index=True)
