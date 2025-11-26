from pydantic import BaseModel


class Species(BaseModel):
    id: int | None
    scientific_name: str | None
    common_name: str | None
