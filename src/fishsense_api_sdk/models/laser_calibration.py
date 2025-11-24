from pydantic import BaseModel


class LaserCalibration(BaseModel):
    id: int | None
