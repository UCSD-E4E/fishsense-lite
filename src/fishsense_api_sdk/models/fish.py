from pydantic import BaseModel


class Fish(BaseModel):
    id: int | None

    species_id: int | None
