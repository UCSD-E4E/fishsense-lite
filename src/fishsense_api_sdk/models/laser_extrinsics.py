from typing import Self

from pydantic import BaseModel


class _LaserCalibration(BaseModel):
    id: int | None


class LaserCalibration:
    @staticmethod
    def _from_internal(internal: _LaserCalibration) -> Self:
        pass

    def _to_internal(self) -> _LaserCalibration:
        pass
