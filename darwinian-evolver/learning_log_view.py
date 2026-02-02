import uuid
from abc import ABC
from abc import abstractmethod

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.population import Population
from darwinian_evolver.problem import Organism


class LearningLogView(ABC):
    """Represents a view into the learning log for a given organism."""

    def __init__(self, population: Population) -> None:
        self._learning_log = population.learning_log
        self._population = population

    @abstractmethod
    def get_entries_for_organism(self, organism: Organism) -> list[LearningLogEntry]:
        """Get the learning log entries visible to the given organism."""
        pass


class EmptyLearningLogView(LearningLogView):
    """A learning log view that provides no entries."""

    def get_entries_for_organism(self, organism: Organism) -> list[LearningLogEntry]:
        return []


class AncestorLearningLogView(LearningLogView):
    """A learning log view that provides entries from the organism's ancestors."""

    _max_depth: int | None

    def __init__(self, population: Population, max_depth: int | None = None) -> None:
        super().__init__(population)
        self._max_depth = max_depth

    def get_entries_for_organism(self, organism: Organism) -> list[LearningLogEntry]:
        entries: list[LearningLogEntry] = []
        current_ancestor = organism
        current_depth = 0
        while current_ancestor is not None and (self._max_depth is None or current_depth <= self._max_depth):
            maybe_entry = self._learning_log.get_entry(current_ancestor.id)
            if maybe_entry is not None:
                entries.append(maybe_entry)
            current_ancestor = current_ancestor.parent
            current_depth += 1

        return entries


class NeighborhoodLearningLogView(LearningLogView):
    """A learning log view that provides entries from organisms within a certain graph distance."""

    _max_distance: int

    def __init__(self, population: Population, max_distance: int) -> None:
        super().__init__(population)
        self._max_distance = max_distance

    def get_entries_for_organism(self, organism: Organism) -> list[LearningLogEntry]:
        # Perform a graph traversal around the given organism.
        visited: set[uuid.UUID] = set()
        organisms_in_range = self._traverse_graph(organism, 0, visited)
        entries: list[LearningLogEntry] = []
        for org in organisms_in_range:
            maybe_entry = self._learning_log.get_entry(org.id)
            if maybe_entry is not None:
                entries.append(maybe_entry)

        return entries

    def _traverse_graph(self, organism: Organism, current_distance: int, visited: set[uuid.UUID]) -> list[Organism]:
        """Perform a recursive graph traversal over the organism graph."""
        if current_distance > self._max_distance:
            return []

        visited.add(organism.id)

        organisms_in_range: list[Organism] = [organism]

        # Traverse to parent
        if organism.parent is not None and organism.parent.id not in visited:
            organisms_in_range.extend(self._traverse_graph(organism.parent, current_distance + 1, visited))

        # Traverse to children
        for child, _ in self._population.get_children(organism):
            if child.id not in visited:
                organisms_in_range.extend(self._traverse_graph(child, current_distance + 1, visited))

        return organisms_in_range
