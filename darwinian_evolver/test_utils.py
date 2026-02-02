"""Common test utilities and helper classes for darwinian_evolver tests."""

from typing import Any
from uuid import uuid4

from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.evolver import Evolver
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.learning_log_view import EmptyLearningLogView
from darwinian_evolver.learning_log_view import LearningLogView
from darwinian_evolver.population import FixedTreePopulation
from darwinian_evolver.population import Population
from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem


class MockOrganism(Organism):
    """A simple mock organism for testing purposes."""


def create_test_organism(
    parent: Organism | None = None,
    score: float = 1.0,
    change_summary: str | None = None,
    num_failures: int = 1,
) -> tuple[MockOrganism, EvaluationResult]:
    """Create a test organism with an evaluation result."""
    organism = MockOrganism(parent=parent, from_change_summary=change_summary)
    evaluation_result = create_test_result(score=score, num_trainable_failures=num_failures)
    return organism, evaluation_result


class MockEvaluator(Evaluator[MockOrganism, EvaluationResult, EvaluationFailureCase]):
    """Simple evaluator for testing."""

    def __init__(self, score: float = 1.0, num_failure_cases: int = 1):
        self.score = score
        self.num_failure_cases = num_failure_cases
        self.evaluate_count = 0
        self.verify_count = 0

    def evaluate(self, organism: MockOrganism) -> EvaluationResult:
        """Evaluate organism and return fixed score with random failure cases."""
        self.evaluate_count += 1
        return create_test_result(score=self.score, num_trainable_failures=self.num_failure_cases)

    def verify_mutation(self, organism: MockOrganism) -> bool:
        """Verify mutation - default implementation passes verification."""
        self.verify_count += 1
        return True


class MockMutator(Mutator[MockOrganism, EvaluationFailureCase]):
    """Simple mutator for testing."""

    def __init__(self, num_children: int = 1, batch_mutation: bool = False):
        super().__init__()
        self.num_children = num_children
        self.batch_mutation = batch_mutation
        self.mutate_count = 0
        self.last_failure_cases = None
        self.last_learning_log_entries = None

    def mutate(
        self,
        organism: MockOrganism,
        failure_cases: list[EvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[MockOrganism]:
        """Mutate organism and return specified number of children."""
        self.mutate_count += 1
        self.last_failure_cases = failure_cases
        self.last_learning_log_entries = learning_log_entries
        return [MockOrganism(parent=organism, from_change_summary=f"mutation_{i}") for i in range(self.num_children)]

    @property
    def supports_batch_mutation(self) -> bool:
        """Return whether this mutator supports batch mutation."""
        return self.batch_mutation


class FailingVerificationEvaluator(MockEvaluator):
    """Evaluator that always fails verification for testing."""

    def verify_mutation(self, organism: MockOrganism) -> bool:
        """Always fail verification."""
        self.verify_count += 1
        return False


# ============================================================================
# Factory Functions
# ============================================================================


def create_mock_problem() -> Problem:
    """Create a simple problem for testing."""
    initial_organism = MockOrganism()
    evaluator = MockEvaluator()
    mutators: list[Mutator] = [MockMutator()]

    return Problem(initial_organism=initial_organism, evaluator=evaluator, mutators=mutators)


def create_test_loop(
    problem: Problem | None = None,
    num_parents_per_iteration: int = 1,
    mutator_concurrency: int = 1,
    evaluator_concurrency: int = 1,
    fixed_children_per_generation: list[int] | None = None,
    sharpness: float | None = None,
    midpoint_score_percentile: float | None = None,
    fixed_midpoint_score: float | None = None,
    batch_size: int = 1,
    should_verify_mutations: bool = False,
    novelty_weight: float | None = None,
    snapshot_to_resume_from: bytes | None = None,
    use_process_pool_executors: bool = False,
) -> EvolveProblemLoop:
    """Create an EvolveProblemLoop with sensible defaults for testing."""

    return EvolveProblemLoop(
        problem=problem or create_mock_problem(),
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=mutator_concurrency,
        evaluator_concurrency=evaluator_concurrency,
        num_parents_per_iteration=num_parents_per_iteration,
        sharpness=sharpness,
        fixed_children_per_generation=fixed_children_per_generation,
        midpoint_score_percentile=midpoint_score_percentile,
        fixed_midpoint_score=fixed_midpoint_score,
        batch_size=batch_size,
        should_verify_mutations=should_verify_mutations,
        novelty_weight=novelty_weight,
        snapshot_to_resume_from=snapshot_to_resume_from,
        use_process_pool_executors=use_process_pool_executors,
    )


def create_weighted_population(
    score: float = 1.0,
    sharpness: float = 10.0,
    fixed_midpoint_score: float | None = 1.0,
    midpoint_score_percentile: float | None = None,
    novelty_weight: float = 1.0,
    parent: MockOrganism | None = None,
    change_summary: str | None = None,
) -> WeightedSamplingPopulation:
    """Create a WeightedSamplingPopulation with sensible defaults for testing."""
    initial_organism, initial_result = create_test_organism(parent=parent, score=score, change_summary=change_summary)
    return WeightedSamplingPopulation(
        initial_organism=initial_organism,
        initial_evaluation_result=initial_result,
        sharpness=sharpness,
        fixed_midpoint_score=fixed_midpoint_score,
        midpoint_score_percentile=midpoint_score_percentile,
        novelty_weight=novelty_weight,
    )


def create_fixed_tree_population(
    score: float = 1.0,
    fixed_children_per_generation: list[int] | None = None,
    parent: MockOrganism | None = None,
    change_summary: str | None = None,
) -> FixedTreePopulation:
    """Create a FixedTreePopulation with sensible defaults for testing."""
    if fixed_children_per_generation is None:
        fixed_children_per_generation = [3, 2]

    initial_organism, initial_result = create_test_organism(parent=parent, score=score, change_summary=change_summary)
    return FixedTreePopulation(
        initial_organism=initial_organism,
        initial_evaluation_result=initial_result,
        fixed_children_per_generation=fixed_children_per_generation,
    )


def create_evolver(
    population: Population | None = None,
    evaluator: Evaluator | None = None,
    mutators: list[Mutator] | None = None,
    learning_log_view_type: tuple[type[LearningLogView], dict[str, Any]] | None = None,
    mutator_concurrency: int = 1,
    evaluator_concurrency: int = 1,
    batch_size: int = 1,
    should_verify_mutations: bool = False,
    use_process_pool_executors: bool = False,
) -> Evolver:
    """Create an Evolver with sensible defaults for testing."""
    if population is None:
        population = create_weighted_population()
    if evaluator is None:
        evaluator = MockEvaluator()
    if mutators is None:
        guaranteed_mutators: list[Mutator] = [MockMutator()]
    else:
        guaranteed_mutators: list[Mutator] = mutators
    if learning_log_view_type is None:
        learning_log_view_type = (EmptyLearningLogView, {})

    return Evolver(
        population=population,
        mutators=guaranteed_mutators,
        evaluator=evaluator,
        learning_log_view_type=learning_log_view_type,
        mutator_concurrency=mutator_concurrency,
        evaluator_concurrency=evaluator_concurrency,
        batch_size=batch_size,
        should_verify_mutations=should_verify_mutations,
        use_process_pool_executors=use_process_pool_executors,
    )


def create_test_failure_case(
    data_point_id: str | None = None, failure_type: str = "test_failure"
) -> EvaluationFailureCase:
    """Create a test failure case with optional custom ID."""
    if data_point_id is None:
        data_point_id = f"fail_{uuid4().hex[:8]}"
    return EvaluationFailureCase(data_point_id=data_point_id, failure_type=failure_type)


def create_test_result(
    score: float = 1.0,
    num_trainable_failures: int = 1,
    num_holdout_failures: int = 0,
    failure_type: str = "test_failure",
    is_viable: bool = True,
) -> EvaluationResult:
    """Create an EvaluationResult with specified number of failure cases."""
    return EvaluationResult(
        score=score,
        trainable_failure_cases=[
            create_test_failure_case(failure_type=failure_type) for _ in range(num_trainable_failures)
        ],
        holdout_failure_cases=[
            create_test_failure_case(failure_type=failure_type) for _ in range(num_holdout_failures)
        ],
        is_viable=is_viable,
    )


def add_test_child(
    population: Population,
    parent: Organism | None = None,
    score: float = 1.5,
    change_summary: str | None = None,
    num_failures: int = 1,
) -> tuple[Organism, EvaluationResult]:
    """Add a child organism to the population and return it."""
    if parent is None:
        parent = population.organisms[0][0]
    child, child_result = create_test_organism(parent=parent, score=score, change_summary=change_summary)
    # Update result with correct number of failures if needed
    if num_failures != 1:
        child_result = create_test_result(score=score, num_trainable_failures=num_failures)
    population.add(child, child_result)
    return child, child_result
