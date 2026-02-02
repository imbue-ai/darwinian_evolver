import uuid

from pydantic import BaseModel
from pydantic import ConfigDict


class LearningLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempted_change: str
    observed_outcome: str


class LearningLog:
    _entry_for_organism: dict[uuid.UUID, LearningLogEntry]

    def __init__(self) -> None:
        self._entry_for_organism = {}

    def add_entry(self, organism_id: uuid.UUID, entry: LearningLogEntry) -> None:
        assert organism_id not in self._entry_for_organism, (
            f"Learning log entry for organism ID {organism_id} already exists."
        )
        self._entry_for_organism[organism_id] = entry

    def get_entry(self, organism_id: uuid.UUID) -> LearningLogEntry | None:
        return self._entry_for_organism.get(organism_id)
