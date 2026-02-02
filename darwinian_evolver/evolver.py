from __future__ import annotations

import concurrent
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pydantic import BaseModel
from pydantic import computed_field

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.learning_log_view import LearningLogView
from darwinian_evolver.population import Population
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism


class EvolverStats(BaseModel):
    num_mutate_calls: int = 0
    num_failure_cases_supplied: int = 0
    num_generated_mutations: int = 0
    num_mutations_after_verification: int = 0
    num_evaluate_calls: int = 0
    num_verify_mutation_calls: int = 0
    num_learning_log_entries_supplied: int = 0

    @computed_field
    def effective_batch_size(self) -> float:
        if self.num_mutate_calls == 0:
            return 0.0
        return self.num_failure_cases_supplied / self.num_mutate_calls

    @computed_field
    def average_learning_log_entries_supplied(self) -> float:
        if self.num_mutate_calls == 0:
            return 0.0
        return self.num_learning_log_entries_supplied / self.num_mutate_calls

    def __add__(self, other: EvolverStats) -> EvolverStats:
        if not isinstance(other, EvolverStats):
            raise TypeError("Can only add EvolverStats")
        return EvolverStats(
            num_mutate_calls=self.num_mutate_calls + other.num_mutate_calls,
            num_failure_cases_supplied=self.num_failure_cases_supplied + other.num_failure_cases_supplied,
            num_generated_mutations=self.num_generated_mutations + other.num_generated_mutations,
            num_mutations_after_verification=self.num_mutations_after_verification
            + other.num_mutations_after_verification,
            num_evaluate_calls=self.num_evaluate_calls + other.num_evaluate_calls,
            num_verify_mutation_calls=self.num_verify_mutation_calls + other.num_verify_mutation_calls,
            num_learning_log_entries_supplied=self.num_learning_log_entries_supplied
            + other.num_learning_log_entries_supplied,
        )

    def __iadd__(self, other: EvolverStats) -> EvolverStats:
        if not isinstance(other, EvolverStats):
            raise TypeError("Can only add EvolverStats")
        self.num_mutate_calls += other.num_mutate_calls
        self.num_failure_cases_supplied += other.num_failure_cases_supplied
        self.num_generated_mutations += other.num_generated_mutations
        self.num_mutations_after_verification += other.num_mutations_after_verification
        self.num_evaluate_calls += other.num_evaluate_calls
        self.num_verify_mutation_calls += other.num_verify_mutation_calls
        self.num_learning_log_entries_supplied += other.num_learning_log_entries_supplied
        return self


OrganismT = TypeVar("OrganismT", bound=Organism)
EvaluationFailureCaseT = TypeVar("EvaluationFailureCaseT", bound=EvaluationFailureCase)


class Evolver:
    _population: Population
    _learning_log_view: LearningLogView
    _mutators: list[Mutator]
    _evaluator: Evaluator
    _mutator_concurrency: int
    _evaluator_concurrency: int
    _use_process_pool_executors: bool
    _batch_size: int
    _should_verify_mutations: bool

    def __init__(
        self,
        # Initial population to start with
        population: Population,
        mutators: list[Mutator],
        evaluator: Evaluator,
        learning_log_view_type: tuple[type[LearningLogView], dict[str, any]],
        mutator_concurrency: int = 10,
        evaluator_concurrency: int = 10,
        # The number of failure cases that we make available to mutators for a given parent organism.
        batch_size: int = 1,
        should_verify_mutations: bool = False,
        use_process_pool_executors: bool = False,
    ) -> None:
        assert mutators, "Mutators list cannot be empty"
        assert mutator_concurrency > 0, "Mutator concurrency must be positive"
        assert evaluator_concurrency > 0, "Evaluator concurrency must be positive"
        assert batch_size > 0, "Batch size must be positive"

        self._population = population
        self._learning_log_view = learning_log_view_type[0](population, **learning_log_view_type[1])
        self._mutators = mutators
        self._evaluator = evaluator
        self._mutator_concurrency = mutator_concurrency
        self._evaluator_concurrency = evaluator_concurrency
        self._use_process_pool_executors = use_process_pool_executors
        self._batch_size = batch_size
        self._should_verify_mutations = should_verify_mutations

    def evolve_iteration(self, num_parents: int, iteration: int | None = None) -> EvolverStats:
        """Evolve the population by generating new organisms."""
        num_mutate_calls = 0
        num_failure_cases_supplied = 0
        num_generated_mutations = 0
        num_mutations_after_verification = 0
        num_evaluate_calls = 0
        num_verify_mutation_calls = 0
        num_learning_log_entries_supplied = 0

        parents = self._population.sample_parents(num_parents, iteration=iteration)

        executor_type = ProcessPoolExecutor if self._use_process_pool_executors else ThreadPoolExecutor
        with (
            executor_type(max_workers=self._mutator_concurrency) as mutator_executor,
            executor_type(max_workers=self._evaluator_concurrency) as evaluator_executor,
        ):
            mutator_futures = []

            for organism, evaluation_result in parents:
                failure_cases = evaluation_result.sample_trainable_failure_cases(batch_size=self._batch_size)
                assert failure_cases, (
                    "sample_parents should only have returned organisms with at least one trainable failure case"
                )
                learning_log_entries = self._learning_log_view.get_entries_for_organism(organism)
                num_learning_log_entries_supplied += len(learning_log_entries)
                for mutator in self._mutators:
                    failure_cases_for_mutator = (
                        failure_cases if mutator.supports_batch_mutation else [failure_cases[0]]
                    )

                    mutator_future = mutator_executor.submit(
                        self._mutate_and_inject_attributes,  # type: ignore[6]
                        organism,
                        mutator,
                        failure_cases_for_mutator,
                        learning_log_entries,
                    )
                    mutator_futures.append(mutator_future)
                    num_mutate_calls += 1
                    num_failure_cases_supplied += len(failure_cases_for_mutator)

            # Build futures that return (organism, passed_verification) tuples
            mutated_organisms_futures = []
            for mutated_organisms_future in concurrent.futures.as_completed(mutator_futures):
                mutated_organisms = mutated_organisms_future.result()
                num_generated_mutations += len(mutated_organisms)
                for mutated_organism in mutated_organisms:
                    if self._should_verify_mutations:
                        future = evaluator_executor.submit(self._verify_mutation, mutated_organism)
                        num_verify_mutation_calls += 1
                    else:
                        # Without verification, all organisms are considered to pass
                        future = concurrent.futures.Future()
                        future.set_result((mutated_organism, True))
                    mutated_organisms_futures.append(future)

            # Then filter by passed_verification boolean
            organism_evaluation_futures = []
            for future in concurrent.futures.as_completed(mutated_organisms_futures):
                organism, should_evaluate = future.result()
                if should_evaluate:
                    num_mutations_after_verification += 1
                    evaluation_future = evaluator_executor.submit(self._evaluator.evaluate, organism)
                    organism_evaluation_futures.append((organism, evaluation_future))
                    num_evaluate_calls += 1
                else:
                    self._population.add_failed_verification(organism)

            # Collect all evaluation results before we add them to the population.
            # This makes sure that population updates are made atomically and organisms from this iteration
            # aren't visible to mutators within the same iteration (including their learning logs).
            concurrent.futures.wait([evaluation_future for _, evaluation_future in organism_evaluation_futures])

            for mutated_organism, evaluation_future in organism_evaluation_futures:
                evaluation_result = evaluation_future.result()
                self._population.add(mutated_organism, evaluation_result)

            return EvolverStats(
                num_mutate_calls=num_mutate_calls,
                num_failure_cases_supplied=num_failure_cases_supplied,
                num_generated_mutations=num_generated_mutations,
                num_mutations_after_verification=num_mutations_after_verification,
                num_evaluate_calls=num_evaluate_calls,
                num_verify_mutation_calls=num_verify_mutation_calls,
                num_learning_log_entries_supplied=num_learning_log_entries_supplied,
            )

    def _verify_mutation(self, organism: OrganismT) -> tuple[OrganismT, bool]:
        return organism, self._evaluator.verify_mutation(organism)

    def _mutate_and_inject_attributes(
        self,
        parent_organism: OrganismT,
        mutator: Mutator[OrganismT, EvaluationFailureCaseT],
        failure_cases: list[EvaluationFailureCaseT],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[OrganismT]:
        mutated_organisms = mutator.mutate(parent_organism, failure_cases, learning_log_entries)
        cast_failure_cases: list[EvaluationFailureCase] = [c for c in failure_cases]
        for organism in mutated_organisms:
            if organism.parent is None:
                organism.parent = parent_organism
            if organism.from_failure_cases is None:
                organism.from_failure_cases = cast_failure_cases
            if organism.from_learning_log_entries is None:
                organism.from_learning_log_entries = learning_log_entries
        return mutated_organisms

    @property
    def population(self) -> Population:
        """Get the current population."""
        return self._population
