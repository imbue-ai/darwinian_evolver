from __future__ import annotations

import math
import pickle
import random
from collections import defaultdict
from uuid import UUID

import numpy as np

from darwinian_evolver.learning_log import LearningLog
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Organism

DEFAULT_PERCENTILES = [float(p) for p in range(0, 101, 5)]


class Population:
    """
    Abstract base class for maintaining a population (archive) of organisms.

    Provides means to sample parents from the population that can be used as a basis for further mutation.
    """

    _organisms: list[tuple[Organism, EvaluationResult]]
    _organisms_by_id: dict[UUID, tuple[Organism, EvaluationResult]]
    _children: defaultdict[UUID, list[UUID]]
    _learning_log: LearningLog
    _organisms_failed_verification: list[Organism]

    def __init__(
        self,
        initial_organism: Organism,
        initial_evaluation_result: EvaluationResult,
    ) -> None:
        """Initialize a Population with an initial organism and its evaluation result."""
        assert initial_organism.parent is None, "Initial organism must not have a parent"
        if not initial_evaluation_result.is_viable:
            raise ValueError(
                f"Initial organism must be viable. Got non-viable evaluation result: {initial_evaluation_result}"
            )

        self._organisms = []
        self._organisms_by_id = {}
        self._children = defaultdict(list)
        self._learning_log = LearningLog()
        self._organisms_failed_verification = []

        self.add(initial_organism, initial_evaluation_result)

    @classmethod
    def from_snapshot(cls, snapshot: bytes) -> Population:
        """Create a Population instance from a snapshot."""
        snapshot_dict = pickle.loads(snapshot)
        if not isinstance(snapshot_dict, dict):
            raise ValueError("Snapshot must be a pickled dictionary")

        # Bypass the public __init__ method
        population = cls.__new__(cls)

        population._organisms = snapshot_dict["organisms"]
        population._organisms_by_id = {
            organism.id: (organism, evaluation_result) for organism, evaluation_result in population._organisms
        }
        population._children = defaultdict(list)
        population._learning_log = LearningLog()
        population._organisms_failed_verification = snapshot_dict.get("organisms_failed_verification", [])
        for organism, evaluation_result in population._organisms:
            population._add_to_learning_log(organism, evaluation_result)
            parent = organism.parent
            if parent is not None:
                population._children[parent.id].append(organism.id)

        return population

    def snapshot(self) -> bytes:
        """
        Create a snapshot of the current population.

        Can be stored to a file and then restored later using `Population.from_snapshot()`.
        """
        # We're using Python pickle instead of Pydantic's JSON serialization because we want to
        # preserve
        # a) the specific types of Organism and EvaluationResult objects
        # b) the parent pointer relationships in the Organism objects
        snapshot = pickle.dumps(
            {
                "class_name": self.__class__.__name__,
                "organisms": self._organisms,
                "organisms_failed_verification": self._organisms_failed_verification,
            }
        )

        return snapshot

    @staticmethod
    def _dump_organism_to_json(organism: Organism) -> dict:
        """Serialize an organism to JSON format."""
        return organism.model_dump(exclude={"parent", "additional_parents"}, serialize_as_any=True, mode="json") | {
            "parent_id": organism.parent.id.hex if organism.parent else None,
            "additional_parent_ids": (
                [additional_parent.id.hex for additional_parent in organism.additional_parents]
                if organism.additional_parents
                else None
            ),
        }

    def log_to_json_dict(self) -> dict:
        """
        Return a JSON representation of the current population.

        This is useful for logging or debugging purposes.
        However, there is currently no functionality to restore a Population from this JSON. Please use `snapshot()` instead for that purpose.
        """
        snapshot_dict = {
            "organisms": [
                {
                    "organism": self._dump_organism_to_json(organism),
                    "evaluation_result": evaluation_result.model_dump(serialize_as_any=True, mode="json"),
                }
                for organism, evaluation_result in self._organisms
            ],
            "organisms_failed_verification": [
                self._dump_organism_to_json(organism) for organism in self._organisms_failed_verification
            ],
        }
        return snapshot_dict

    def add(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
        """Add an organism to the population with its evaluation result."""
        assert organism.id not in self._organisms_by_id, f"Organism with ID {organism.id} is already in the population"
        self._organisms.append((organism, evaluation_result))
        self._organisms_by_id[organism.id] = (organism, evaluation_result)
        self._add_to_learning_log(organism, evaluation_result)
        parent = organism.parent
        if parent is not None:
            self._children[parent.id].append(organism.id)

    def add_failed_verification(self, organism: Organism) -> None:
        """Add an organism that failed verification."""
        self._organisms_failed_verification.append(organism)

    def sample_parents(
        self, k: int, iteration: int | None = None, replace: bool = True, novelty_weight: float | None = None
    ) -> list[tuple[Organism, EvaluationResult]]:
        """Sample k parents from the population.

        Args:
            k: Number of parents to sample
            iteration: Optional iteration number (used by some subclasses)
            replace: If True, sample with replacement. If False, sample without replacement.
            novelty_weight: Optional weighting factor for novelty bonus.

        Returns:
            List of (organism, evaluation_result) tuples representing the sampled parents.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement sample_parents")

    def get_best(self) -> tuple[Organism, EvaluationResult]:
        """Get the highest-scoring organism in the population."""
        best_organism = None
        for organism, evaluation_result in self._organisms:
            if best_organism is None or evaluation_result.score > best_organism[1].score:
                best_organism = (organism, evaluation_result)

        assert best_organism is not None, "Population was unexpectedly empty"
        return best_organism

    def get_score_percentiles(self, percentiles: list[float] = DEFAULT_PERCENTILES) -> dict[float, float]:
        """Get the score percentiles over the population."""
        scores = [evaluation_result.score for _, evaluation_result in self._organisms]
        if not scores:
            return {percentile: 0.0 for percentile in percentiles}

        # Compute the percentiles
        scores.sort()
        n = len(scores)
        score_percentiles = {}
        for percentile in percentiles:
            if n == 1:
                score_percentiles[percentile] = scores[0]
                continue

            # Calculate the score using linear interpolation between two indices
            k = (n - 1) * (percentile / 100.0)
            f = math.floor(k)
            c = math.ceil(k)

            if f == c:
                score_percentiles[percentile] = scores[int(k)]
            else:
                d0 = scores[int(f)] * (c - k)
                d1 = scores[int(c)] * (k - f)
                score_percentiles[percentile] = d0 + d1

        return score_percentiles

    @property
    def organisms(self) -> list[tuple[Organism, EvaluationResult]]:
        """Get the full list of organisms in the population."""
        return self._organisms

    def get_children(self, parent: Organism) -> list[tuple[Organism, EvaluationResult]]:
        """Get all children of the parent organism."""
        return [self._organisms_by_id[child_id] for child_id in self._children[parent.id]]

    @property
    def learning_log(self) -> LearningLog:
        """Get the learning log associated with this population."""
        return self._learning_log

    def _add_to_learning_log(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
        attempted_change = organism.from_change_summary
        if attempted_change is None:
            return

        if organism.parent is not None:
            parent_result = self._organisms_by_id[organism.parent.id][1]
        else:
            parent_result = None

        observed_outcome = evaluation_result.format_observed_outcome(parent_result)

        entry = LearningLogEntry(attempted_change=attempted_change, observed_outcome=observed_outcome)
        self._learning_log.add_entry(organism.id, entry)


class WeightedSamplingPopulation(Population):
    """
    A population implementation that uses weighted sampling for parent selection.

    The sampling approach is based on Zhang et al., 2025, "Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents".
    """

    _sharpness: float
    _fixed_midpoint_score: float | None
    _midpoint_score_percentile: float | None
    _novelty_weight: float

    def __init__(
        self,
        initial_organism: Organism,
        initial_evaluation_result: EvaluationResult,
        sharpness: float = 10.0,
        fixed_midpoint_score: float | None = None,
        midpoint_score_percentile: float | None = 75.0,
        novelty_weight: float = 1.0,
    ) -> None:
        """Initialize a WeightedSamplingPopulation with an initial organism and its evaluation result."""
        assert sharpness > 0, "Sharpness must be positive"
        assert (fixed_midpoint_score is None) != (midpoint_score_percentile is None), (
            f"Exactly one of fixed_midpoint_score or midpoint_score_percentile must be set (got {fixed_midpoint_score}, {midpoint_score_percentile})"
        )
        assert novelty_weight >= 0, "Novelty weight must be non-negative"

        super().__init__(initial_organism, initial_evaluation_result)

        self.set_sharpness(sharpness)
        self.set_fixed_midpoint_score(fixed_midpoint_score)
        self.set_midpoint_score_percentile(midpoint_score_percentile)
        self.set_novelty_weight(novelty_weight)

    def set_sharpness(self, sharpness: float) -> None:
        assert sharpness > 0, "Sharpness must be positive"
        self._sharpness = sharpness

    def set_fixed_midpoint_score(self, fixed_midpoint_score: float | None) -> None:
        self._fixed_midpoint_score = fixed_midpoint_score
        if fixed_midpoint_score is not None:
            self._midpoint_score_percentile = None

    def set_midpoint_score_percentile(self, midpoint_score_percentile: float | None) -> None:
        if midpoint_score_percentile is not None and not (0 <= midpoint_score_percentile <= 100):
            raise ValueError(f"midpoint_score_percentile must be between 0 and 100, got {midpoint_score_percentile}")
        self._midpoint_score_percentile = midpoint_score_percentile
        if midpoint_score_percentile is not None:
            self._fixed_midpoint_score = None

    def set_novelty_weight(self, novelty_weight: float) -> None:
        assert novelty_weight >= 0, "Novelty weight must be non-negative"
        self._novelty_weight = novelty_weight

    @classmethod
    def from_snapshot(cls, snapshot: bytes) -> WeightedSamplingPopulation:
        """Create a WeightedSamplingPopulation instance from a snapshot."""
        population = super().from_snapshot(snapshot)

        snapshot_dict = pickle.loads(snapshot)
        population._sharpness = snapshot_dict["sharpness"]
        population._fixed_midpoint_score = snapshot_dict["fixed_midpoint_score"]
        population._midpoint_score_percentile = snapshot_dict["midpoint_score_percentile"]
        population._novelty_weight = snapshot_dict.get("novelty_weight", 1.0)

        return population

    def snapshot(self) -> bytes:
        """Create a snapshot of the current WeightedSamplingPopulation."""
        snapshot_dict = pickle.loads(super().snapshot())
        snapshot_dict["sharpness"] = self._sharpness
        snapshot_dict["novelty_weight"] = self._novelty_weight
        snapshot_dict["fixed_midpoint_score"] = self._fixed_midpoint_score
        snapshot_dict["midpoint_score_percentile"] = self._midpoint_score_percentile
        return pickle.dumps(snapshot_dict)

    def sample_parents(
        self, k: int, iteration: int | None = None, replace: bool = True, novelty_weight: float | None = None
    ) -> list[tuple[Organism, EvaluationResult]]:
        """Sample k parents from the population using weighted sampling.

        Args:
            k: Number of parents to sample
            iteration: Optional iteration number (unused by this implementation)
            replace: If True, sample with replacement. If False, sample without replacement.
            novelty_weight: Optional weighting factor for novelty bonus. If None, uses the population's configured novelty weight.
        """
        # To be eligible for parent selection, an organism must:
        # * have failed in at least one trainable evaluation task
        # * be viable
        eligible_organisms = [
            (organism, evaluation_result)
            for organism, evaluation_result in self._organisms
            if evaluation_result.is_viable and len(evaluation_result.trainable_failure_cases) > 0
        ]
        if not eligible_organisms:
            raise RuntimeError("No eligible organisms for parent selection")

        if novelty_weight is None:
            novelty_weight = self._novelty_weight
        weights = self._compute_weights(eligible_organisms, novelty_weight)

        if replace:
            return random.choices(eligible_organisms, weights=weights, k=k)
        else:
            if k > len(eligible_organisms):
                raise ValueError(
                    f"Cannot sample {k} parents without replacement from {len(eligible_organisms)} eligible organisms"
                )
            probabilities = np.array(weights) / sum(weights)
            indices = np.random.choice(len(eligible_organisms), size=k, replace=False, p=probabilities)
            return [eligible_organisms[i] for i in indices]

    def _compute_weights(
        self, eligible_organisms: list[tuple[Organism, EvaluationResult]], novelty_weight: float
    ) -> list[float]:
        """Implements weighting according to section "A.2 Parent Selection" from Zhang et al. 2025."""
        midpoint_score = self._compute_midpoint_score()
        weights = []
        for organism, evaluation_result in eligible_organisms:
            sigmoid_performance = self._compute_sigmoid_performance(evaluation_result, midpoint_score=midpoint_score)
            novelty_bonus = self._compute_novelty_bonus(organism, novelty_weight)
            weight = sigmoid_performance * novelty_bonus

            assert weight >= 0
            weights.append(weight)

        return weights

    def _compute_midpoint_score(self) -> float:
        midpoint_score_percentile = self._midpoint_score_percentile
        if midpoint_score_percentile is not None:
            # Use a dynamic midpoint score based on the current score distribution in the population
            return self.get_score_percentiles([midpoint_score_percentile])[midpoint_score_percentile]
        else:
            assert self._fixed_midpoint_score is not None
            return self._fixed_midpoint_score

    def _compute_sigmoid_performance(self, evaluation_result: EvaluationResult, midpoint_score: float) -> float:
        """Compute the sigmoid-scaled performance of an evaluation result."""
        sigmoid_performance = 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))
        return sigmoid_performance

    def _compute_novelty_bonus(self, organism: Organism, novelty_weight: float) -> float:
        """
        Compute the novelty bonus based on the number of children.

        This assigns a bonus to organisms that haven't been explored as much, encouraging diversity in the population.
        """
        num_children = len(self._children[organism.id])
        novelty_bonus = 1 / (1 + novelty_weight * num_children)
        return novelty_bonus


class FixedTreePopulation(Population):
    """
    A population variant that generates a fixed tree structure.

    Instead of weighted sampling, each organism in generation g produces exactly n children,
    where n is determined by the fixed_children_per_generation pattern for that generation.
    """

    def __init__(
        self,
        initial_organism: Organism,
        initial_evaluation_result: EvaluationResult,
        fixed_children_per_generation: list[int] | None = None,
    ) -> None:
        """Initialize a FixedTreePopulation with a children pattern."""
        if fixed_children_per_generation is None:
            raise ValueError("fixed_children_per_generation is required for FixedTreePopulation")
        assert fixed_children_per_generation, "fixed_children_per_generation must be non-empty"
        assert all(n > 0 for n in fixed_children_per_generation), "All child counts must be positive"

        super().__init__(initial_organism, initial_evaluation_result)
        self._fixed_children_per_generation = fixed_children_per_generation

    @classmethod
    def from_snapshot(cls, snapshot: bytes) -> FixedTreePopulation:
        """Create a FixedTreePopulation instance from a snapshot."""
        population = super().from_snapshot(snapshot)

        snapshot_dict = pickle.loads(snapshot)
        population._fixed_children_per_generation = snapshot_dict.get("fixed_children_per_generation")
        if population._fixed_children_per_generation is None:
            raise ValueError("FixedTreePopulation snapshot missing fixed_children_per_generation")

        return population

    def snapshot(self) -> bytes:
        """Create a snapshot of the current FixedTreePopulation."""
        snapshot_dict = pickle.loads(super().snapshot())
        snapshot_dict["fixed_children_per_generation"] = self._fixed_children_per_generation
        return pickle.dumps(snapshot_dict)

    def sample_parents(
        self, k: int, iteration: int | None = None, replace: bool = True, novelty_weight: float | None = None
    ) -> list[tuple[Organism, EvaluationResult]]:
        """
        Select all organisms from the current generation frontier, each repeated n times.

        Args:
            k: Ignored in tree mode
            iteration: Required. Used to determine number of children per parent from the pattern.
            replace: Unused by this implementation
            novelty_weight: Unused by this implementation

        Returns:
            List of (organism, evaluation_result) tuples, with each frontier organism repeated
            according to the fixed_children_per_generation pattern.
        """
        if iteration is None:
            raise ValueError("FixedTreePopulation requires iteration parameter")

        # Get number of children per parent for this iteration
        num_children_per_parent = self._fixed_children_per_generation[
            iteration % len(self._fixed_children_per_generation)
        ]

        # Get all organisms from the current generation frontier
        current_generation = self._get_current_generation_frontier()

        if not current_generation:
            raise RuntimeError("No organisms in current generation frontier")

        # Each organism produces num_children_per_parent children
        parents = []
        for parent in current_generation:
            parents.extend([parent] * num_children_per_parent)

        return parents

    def _get_current_generation_frontier(self) -> list[tuple[Organism, EvaluationResult]]:
        """Get all organisms from the most recent generation."""
        if not self._organisms:
            return []

        max_gen = max(self._compute_generation(org) for org, _ in self._organisms)
        return [(org, result) for org, result in self._organisms if self._compute_generation(org) == max_gen]

    @staticmethod
    def _compute_generation(organism: Organism) -> int:
        """Compute the generation number of an organism by counting ancestors."""
        gen = 0
        current = organism
        while current.parent is not None:
            gen += 1
            current = current.parent
        return gen
