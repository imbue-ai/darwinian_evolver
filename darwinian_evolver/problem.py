from __future__ import annotations

import random
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Generic
from typing import Self
from typing import TypeVar
from uuid import UUID
from uuid import uuid4

if TYPE_CHECKING:
    from darwinian_evolver.population import Population

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field
from pydantic import model_validator

from darwinian_evolver.learning_log import LearningLogEntry

# TypeVars for generic typing support
OrganismT = TypeVar("OrganismT", bound="Organism")
EvaluationResultT = TypeVar("EvaluationResultT", bound="EvaluationResult")
EvaluationFailureCaseT = TypeVar("EvaluationFailureCaseT", bound="EvaluationFailureCase")


class Organism(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        description=" ".join(
            [
                "Unique identifier for the organism.",
                "Useful for reconstructing lineages when exporting a population of organisms to a format that doesn't maintain object identity (such as JSON).",
            ]
        ),
        frozen=True,
    )

    parent: Organism | None = None
    additional_parents: list[Organism] = Field(default_factory=list)
    # If this organism was the result of a mutation, these fields will be populated with the failure cases and learning logs that informed the mutation.
    from_failure_cases: list[EvaluationFailureCase] | None = None
    from_learning_log_entries: list[LearningLogEntry] | None = None
    # Note: Mutators are not required to set from_change_summary. So it can be None even if the organism was the result of a mutation.
    from_change_summary: str | None = Field(
        default=None,
        description="A summary of the change that was made to build this organism from its parent. Used for addition to the learning log.",
    )

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        """Additional properties that should be included in the organism info and tooltip."""
        return {}


class EvaluationFailureCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_point_id: str = Field(
        description="An identifier for the data point on which the organism failed. Typically used by the evaluator's verify_mutation function to know which data points it needs to re-evaluate following a mutation."
    )
    failure_type: str = Field(
        description="Type of failure that occurred. Used for grouping similar failure cases for mini batches and for weighted sampling.",
        default="default",
    )


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(description="Overall score of the organism. Used for sampling parents.")

    trainable_failure_cases: list[EvaluationFailureCase] = Field(
        description="Failure cases that can be used to inform a future mutation."
    )
    holdout_failure_cases: list[EvaluationFailureCase] = Field(
        default_factory=list,
        description="Holdout failure cases are never passed to mutators, but can still affect the score of the organism.",
    )

    is_viable: bool = Field(default=True, description="Non-viable organisms will not be considered as parents.")

    def format_observed_outcome(self, parent_result: EvaluationResult | None, ndigits: int = 2) -> str:
        """
        Return a string summarizing the observed outcome of the change that was made from the parent.

        Used for addition to the learning log.

        This is a default implementation that can be overridden by subclasses to include more specific information.
        """
        if not self.is_viable:
            return "Inconclusive - the resulting organism was not viable."
        else:
            rounded_score = round(self.score, ndigits)
            outcome = f"The organism achieved an overall fitness score of {rounded_score}."
            if parent_result is not None:
                rounded_parent_score = round(parent_result.score, ndigits)
                if rounded_score > rounded_parent_score:
                    outcome += f" This was an improvement over the parent's score of {rounded_parent_score}."
                elif rounded_score < rounded_parent_score:
                    outcome += f" This was worse than the parent's score of {rounded_parent_score}."
                else:
                    outcome += f" This was the same as the parent's score of {rounded_parent_score}."

            return outcome

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        """Additional properties that should be included in the organism evaluation info and tooltip."""
        return {}

    def sample_trainable_failure_cases(self, batch_size: int = 1) -> list[EvaluationFailureCase]:
        """
        Sample a batch of failure cases from the evaluation result for mutation purposes.

        All failure cases in the resulting batch will be of the same failure_type.
        """
        if not self.trainable_failure_cases:
            return []

        # We use a two-stage approach for sampling. First, we sample a failure type. Then we sample
        # from the failure cases of that type.
        failure_type_frequencies = defaultdict(float)
        for failure_case in self.trainable_failure_cases:
            failure_type_frequencies[failure_case.failure_type] += 1

        # Apply any custom failure type weights
        for failure_type, weight in self.failure_type_weights.items():
            assert weight > 0.0, "Failure type weights must be strictly positive"
            if failure_type in failure_type_frequencies:
                failure_type_frequencies[failure_type] *= weight

        # Now sample a failure type proportional to its (weighted) frequency
        failure_type = random.choices(
            list(failure_type_frequencies.keys()),
            weights=list(failure_type_frequencies.values()),
            k=1,
        )[0]

        # Now sample up to batch_size failure cases of that type
        failure_cases_of_type = [
            failure_case for failure_case in self.trainable_failure_cases if failure_case.failure_type == failure_type
        ]
        effective_batch_size = min(batch_size, len(failure_cases_of_type))
        return random.sample(failure_cases_of_type, effective_batch_size)

    @property
    def failure_cases(self) -> list[EvaluationFailureCase]:
        """Get all failure cases, both trainable and holdout."""
        return self.trainable_failure_cases + self.holdout_failure_cases

    @property
    def failure_type_weights(self) -> dict[str, float]:
        """
        Can be overridden to provide custom sampling weights for each failure type.

        The default weight for failure types not included in the returned dictionary is 1.0.
        """
        return {}


class MutatorContext:
    def __init__(self, population: "Population") -> None:
        self._population = population

    @property
    def population(self) -> "Population":
        return self._population


class Mutator(ABC, Generic[OrganismT, EvaluationFailureCaseT]):
    """A mutator is a class that can mutate an organism."""

    def __init__(self) -> None:
        self._context: MutatorContext | None = None

    def set_context(self, context: MutatorContext) -> None:
        self._context = context

    @abstractmethod
    def mutate(
        self,
        organism: OrganismT,
        failure_cases: list[EvaluationFailureCaseT],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[OrganismT]:
        """
        Generate mutated offspring of the organism with a goal of improving on the given failure cases.

        The failure_cases list will have size exactly 1 if supports_batch_mutation is False, or at least 1
        if supports_batch_mutation is True.

        Mutators can return zero, one, or multiple mutated organisms.
        """
        raise NotImplementedError("Mutators must implement the mutate method")

    @property
    def supports_batch_mutation(self) -> bool:
        """
        Whether the mutator supports batch mutation.

        If True, the `mutate` method can accept multiple failure cases at once.
        """
        return False


class Evaluator(ABC, Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    _output_dir: str | None = None
    _s3_dir: str | None = None

    @abstractmethod
    def evaluate(self, organism: OrganismT) -> EvaluationResultT:
        """Evaluate the organism and return a fitness score."""
        raise NotImplementedError("Evaluators must implement the evaluate method")

    def verify_mutation(self, organism: OrganismT) -> bool:
        """
        Verify that the mutation of the organism has addressed the given failure cases (or some fraction of them).

        This method is optional for evaluators to implement.
        """
        raise NotImplementedError("This evaluator does not support mutation verification")

    def set_output_dir(self, output_dir: str) -> None:
        """Set the output directory where evaluator-specific outputs should be written."""
        self._output_dir = output_dir

    def set_s3_dir(self, s3_dir: str) -> None:
        """Set the S3 directory where evaluator-specific outputs should be uploaded."""
        self._s3_dir = s3_dir


class Problem(BaseModel, Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    initial_organism: OrganismT
    evaluator: Evaluator[OrganismT, EvaluationResultT, EvaluationFailureCaseT]
    mutators: list[Mutator[OrganismT, EvaluationFailureCaseT]]

    @model_validator(mode="after")  # pyre-ignore[56]
    def check_initial_organism(self) -> Self:
        if self.initial_organism.parent is not None:
            raise ValueError("Initial organism should not have a parent")
        return self

    @model_validator(mode="after")  # pyre-ignore[56]
    def check_mutators(self) -> Self:
        if not self.mutators:
            raise ValueError("At least one mutator is required")
        return self
