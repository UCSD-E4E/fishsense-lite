from pydantic import BaseModel


class Measurement(BaseModel):
    id: int | None
    length_m: float | None

    image_id: int | None
    fish_id: int | None
