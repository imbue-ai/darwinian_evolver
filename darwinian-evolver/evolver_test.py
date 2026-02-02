"""Tests for Evolver and EvolverStats classes."""

from darwinian_evolver.evolver import Evolver
from darwinian_evolver.evolver import EvolverStats
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.learning_log_view import EmptyLearningLogView
from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.test_utils import FailingVerificationEvaluator
from darwinian_evolver.test_utils import MockEvaluator
from darwinian_evolver.test_utils import MockMutator
from darwinian_evolver.test_utils import create_evolver
from darwinian_evolver.test_utils import create_weighted_population

# ============================================================================
# EvolverStats Tests
# ============================================================================


def assert_evolver_stats(
    stats: EvolverStats,
    *,
    num_mutate_calls: int | None = None,
    num_failure_cases_supplied: int | None = None,
    num_generated_mutations: int | None = None,
    num_mutations_after_verification: int | None = None,
    num_evaluate_calls: int | None = None,
    num_verify_mutation_calls: int | None = None,
    num_learning_log_entries_supplied: int | None = None,
) -> None:
    """Assert EvolverStats fields match expected values (only checks non-None fields)."""
    if num_mutate_calls is not None:
        assert stats.num_mutate_calls == num_mutate_calls
    if num_failure_cases_supplied is not None:
        assert stats.num_failure_cases_supplied == num_failure_cases_supplied
    if num_generated_mutations is not None:
        assert stats.num_generated_mutations == num_generated_mutations
    if num_mutations_after_verification is not None:
        assert stats.num_mutations_after_verification == num_mutations_after_verification
    if num_evaluate_calls is not None:
        assert stats.num_evaluate_calls == num_evaluate_calls
    if num_verify_mutation_calls is not None:
        assert stats.num_verify_mutation_calls == num_verify_mutation_calls
    if num_learning_log_entries_supplied is not None:
        assert stats.num_learning_log_entries_supplied == num_learning_log_entries_supplied


def test_evolver_stats_initialization() -> None:
    """Test EvolverStats initialization with default values."""
    stats = EvolverStats()
    assert stats.num_mutate_calls == 0
    assert stats.num_failure_cases_supplied == 0
    assert stats.num_generated_mutations == 0
    assert stats.num_mutations_after_verification == 0
    assert stats.num_evaluate_calls == 0
    assert stats.num_verify_mutation_calls == 0
    assert stats.num_learning_log_entries_supplied == 0


def test_evolver_stats_addition() -> None:
    """Test adding two EvolverStats objects."""
    stats1 = EvolverStats(
        num_mutate_calls=5,
        num_failure_cases_supplied=10,
        num_generated_mutations=15,
        num_mutations_after_verification=12,
        num_evaluate_calls=12,
        num_verify_mutation_calls=15,
        num_learning_log_entries_supplied=20,
    )
    stats2 = EvolverStats(
        num_mutate_calls=3,
        num_failure_cases_supplied=6,
        num_generated_mutations=9,
        num_mutations_after_verification=8,
        num_evaluate_calls=8,
        num_verify_mutation_calls=9,
        num_learning_log_entries_supplied=12,
    )

    result = stats1 + stats2

    assert_evolver_stats(
        result,
        num_mutate_calls=8,
        num_failure_cases_supplied=16,
        num_generated_mutations=24,
        num_mutations_after_verification=20,
        num_evaluate_calls=20,
        num_verify_mutation_calls=24,
        num_learning_log_entries_supplied=32,
    )


def test_evolver_stats_in_place_addition() -> None:
    """Test += operator for EvolverStats."""
    stats1 = EvolverStats(
        num_mutate_calls=5,
        num_failure_cases_supplied=10,
        num_generated_mutations=15,
        num_mutations_after_verification=12,
        num_evaluate_calls=12,
        num_verify_mutation_calls=15,
        num_learning_log_entries_supplied=20,
    )
    stats2 = EvolverStats(
        num_mutate_calls=3,
        num_failure_cases_supplied=6,
        num_generated_mutations=9,
        num_mutations_after_verification=8,
        num_evaluate_calls=8,
        num_verify_mutation_calls=9,
        num_learning_log_entries_supplied=12,
    )

    stats1 += stats2

    assert_evolver_stats(
        stats1,
        num_mutate_calls=8,
        num_failure_cases_supplied=16,
        num_generated_mutations=24,
        num_mutations_after_verification=20,
        num_evaluate_calls=20,
        num_verify_mutation_calls=24,
        num_learning_log_entries_supplied=32,
    )


def test_evolver_stats_effective_batch_size() -> None:
    """Test effective_batch_size computed field."""
    stats = EvolverStats(num_mutate_calls=10, num_failure_cases_supplied=30)
    assert stats.effective_batch_size == 3.0


def test_evolver_stats_effective_batch_size_zero_calls() -> None:
    """Test effective_batch_size with zero mutate calls."""
    stats = EvolverStats(num_mutate_calls=0, num_failure_cases_supplied=0)
    assert stats.effective_batch_size == 0.0


def test_evolver_stats_effective_batch_size_fractional() -> None:
    """Test effective_batch_size with fractional result."""
    stats = EvolverStats(num_mutate_calls=3, num_failure_cases_supplied=10)
    assert abs(stats.effective_batch_size - 3.333333) < 0.00001


def test_evolver_stats_average_learning_log_entries() -> None:
    """Test average_learning_log_entries_supplied computed field."""
    stats = EvolverStats(num_mutate_calls=5, num_learning_log_entries_supplied=15)
    assert stats.average_learning_log_entries_supplied == 3.0


def test_evolver_stats_average_learning_log_entries_zero_calls() -> None:
    """Test average_learning_log_entries_supplied with zero mutate calls."""
    stats = EvolverStats(num_mutate_calls=0, num_learning_log_entries_supplied=0)
    assert stats.average_learning_log_entries_supplied == 0.0


# ============================================================================
# Evolver Initialization Tests
# ============================================================================


def test_evolver_initialization() -> None:
    """Test Evolver initialization with valid parameters."""
    population = create_weighted_population()
    evaluator = MockEvaluator()
    mutators: list[Mutator] = [MockMutator()]

    evolver = create_evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        mutator_concurrency=5,
        evaluator_concurrency=5,
        batch_size=1,
    )

    assert evolver.population == population
    assert evolver._mutators == mutators
    assert evolver._evaluator == evaluator
    assert evolver._mutator_concurrency == 5
    assert evolver._evaluator_concurrency == 5
    assert evolver._batch_size == 1


# ============================================================================
# Evolver Evolution Tests
# ============================================================================


def test_evolve_iteration_single_parent() -> None:
    """Test single iteration with one parent."""
    population = create_weighted_population(fixed_midpoint_score=0.5)
    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutator: Mutator = MockMutator(num_children=1)

    evolver = create_evolver(
        population=population,
        mutators=[mutator],
        evaluator=evaluator,
    )

    # Population starts with 1 organism
    assert len(population.organisms) == 1

    stats = evolver.evolve_iteration(num_parents=1)

    # After evolution, we should have 2 organisms (initial + 1 child)
    assert len(population.organisms) == 2
    assert_evolver_stats(
        stats,
        num_mutate_calls=1,
        num_failure_cases_supplied=1,
        num_generated_mutations=1,
        num_evaluate_calls=1,
    )
    assert mutator.mutate_count == 1
    assert evaluator.evaluate_count == 1


def test_evolve_iteration_multiple_parents() -> None:
    """Test iteration with multiple parents."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [MockMutator(num_children=1)]

    evolver = create_evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        mutator_concurrency=2,
        evaluator_concurrency=2,
    )

    stats = evolver.evolve_iteration(num_parents=3)

    # 3 parents * 1 mutator * 1 child = 3 new organisms
    assert len(population.organisms) == 4  # initial + 3 children
    assert stats.num_mutate_calls == 3
    assert stats.num_generated_mutations == 3
    assert stats.num_evaluate_calls == 3


def test_evolve_iteration_multiple_mutators() -> None:
    """Test iteration with multiple mutators."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutator1 = MockMutator(num_children=1)
    mutator2 = MockMutator(num_children=2)
    mutators: list[Mutator] = [mutator1, mutator2]

    evolver = create_evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        mutator_concurrency=4,
        evaluator_concurrency=4,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # 1 parent * 2 mutators: first produces 1 child, second produces 2 children = 3 total
    assert len(population.organisms) == 4  # initial + 3 children
    assert stats.num_mutate_calls == 2  # 1 parent * 2 mutators
    assert stats.num_generated_mutations == 3  # 1 + 2 children
    assert stats.num_evaluate_calls == 3


def test_evolve_iteration_with_verification() -> None:
    """Test mutation verification flow."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [MockMutator(num_children=2)]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        should_verify_mutations=True,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # All mutations should pass verification
    assert stats.num_generated_mutations == 2
    assert stats.num_verify_mutation_calls == 2
    assert stats.num_mutations_after_verification == 2
    assert stats.num_evaluate_calls == 2
    assert evaluator.verify_count == 2
    assert len(population.organisms) == 3  # initial + 2 children


def test_evolve_iteration_verification_failures() -> None:
    """Test that failed verification prevents evaluation."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = FailingVerificationEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [MockMutator(num_children=3)]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        should_verify_mutations=True,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # All mutations fail verification
    assert stats.num_generated_mutations == 3
    assert stats.num_verify_mutation_calls == 3
    assert stats.num_mutations_after_verification == 0  # None passed verification
    assert stats.num_evaluate_calls == 0  # None were evaluated
    assert len(population.organisms) == 1  # Only initial organism
    assert len(population._organisms_failed_verification) == 3


def test_evolve_iteration_without_verification() -> None:
    """Test that without verification, all mutations are evaluated."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [MockMutator(num_children=2)]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        should_verify_mutations=False,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # Without verification, all mutations go directly to evaluation
    assert stats.num_generated_mutations == 2
    assert stats.num_verify_mutation_calls == 0
    assert stats.num_mutations_after_verification == 2
    assert stats.num_evaluate_calls == 2
    assert evaluator.verify_count == 0


def test_evolve_iteration_batch_mutation() -> None:
    """Test with batch-supporting mutator."""
    # Create an organism with multiple failure cases so batch sampling works
    initial_organism = Organism()
    initial_result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id=f"fail_{i}", failure_type="test_failure") for i in range(5)
        ],
        is_viable=True,
    )
    population = WeightedSamplingPopulation(
        initial_organism=initial_organism,
        initial_evaluation_result=initial_result,
        sharpness=10.0,
        fixed_midpoint_score=0.5,
        midpoint_score_percentile=None,
    )

    evaluator = MockEvaluator(score=1.5, num_failure_cases=5)
    batch_mutator = MockMutator(num_children=1, batch_mutation=True)
    mutators: list[Mutator] = [batch_mutator]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        batch_size=3,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # Batch mutator should receive 3 failure cases (batch_size=3)
    assert stats.num_mutate_calls == 1
    assert stats.num_failure_cases_supplied == 3
    assert batch_mutator.last_failure_cases is not None
    assert len(batch_mutator.last_failure_cases) == 3


def test_evolve_iteration_non_batch_mutator_with_batch_size() -> None:
    """Test that non-batch mutators only receive 1 failure case regardless of batch_size."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=5)
    non_batch_mutator = MockMutator(num_children=1, batch_mutation=False)
    mutators: list[Mutator] = [non_batch_mutator]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        batch_size=3,
    )

    stats = evolver.evolve_iteration(num_parents=1)

    # Non-batch mutator should only receive 1 failure case
    assert stats.num_mutate_calls == 1
    assert stats.num_failure_cases_supplied == 1
    assert non_batch_mutator.last_failure_cases is not None
    assert len(non_batch_mutator.last_failure_cases) == 1


def test_evolve_iteration_stats_tracking() -> None:
    """Test that stats are correctly accumulated across multiple operations."""
    # Create an organism with multiple failure cases so batch sampling works
    initial_organism = Organism()
    initial_result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id=f"fail_{i}", failure_type="test_failure") for i in range(5)
        ],
        is_viable=True,
    )
    population = WeightedSamplingPopulation(
        initial_organism=initial_organism,
        initial_evaluation_result=initial_result,
        sharpness=10.0,
        fixed_midpoint_score=0.5,
        midpoint_score_percentile=None,
    )

    evaluator = MockEvaluator(score=1.5, num_failure_cases=5)
    mutator1 = MockMutator(num_children=2, batch_mutation=True)
    mutator2 = MockMutator(num_children=1, batch_mutation=False)
    mutators: list[Mutator] = [mutator1, mutator2]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=4,
        evaluator_concurrency=4,
        batch_size=2,
    )

    stats = evolver.evolve_iteration(num_parents=2)

    # 2 parents * 2 mutators = 4 mutate calls
    assert stats.num_mutate_calls == 4

    # mutator1 gets 2 failure cases (batch), mutator2 gets 1 (non-batch), for each of 2 parents
    assert stats.num_failure_cases_supplied == 2 * (2 + 1)

    # mutator1 produces 2 children, mutator2 produces 1, for each of 2 parents
    assert stats.num_generated_mutations == 2 * (2 + 1)

    # All mutations are evaluated
    assert stats.num_evaluate_calls == 2 * (2 + 1)


def test_mutate_and_inject_attributes() -> None:
    """Test that organism attributes are correctly injected after mutation."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [MockMutator(num_children=1)]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
    )

    # Run evolution to trigger attribute injection
    evolver.evolve_iteration(num_parents=1)

    # Get the mutated organism
    mutated_organism = population.organisms[1][0]
    initial_organism = population.organisms[0][0]

    # Verify attributes were injected
    assert mutated_organism.parent == initial_organism
    assert mutated_organism.from_failure_cases is not None
    assert len(mutated_organism.from_failure_cases) == 1
    assert mutated_organism.from_learning_log_entries is not None
    assert mutated_organism.from_change_summary == "mutation_0"


def test_mutate_and_inject_attributes_preserves_existing() -> None:
    """Test that attribute injection doesn't overwrite existing attributes."""

    class CustomMutator(Mutator):
        """Mutator that sets custom parent and failure cases."""

        def mutate(
            self,
            organism: Organism,
            failure_cases: list[EvaluationFailureCase],
            learning_log_entries: list[LearningLogEntry],
        ) -> list[Organism]:
            custom_parent = Organism()
            custom_failure = EvaluationFailureCase(data_point_id="custom", failure_type="custom")
            custom_learning = LearningLogEntry(attempted_change="custom", observed_outcome="custom")

            return [
                Organism(
                    parent=custom_parent,
                    from_failure_cases=[custom_failure],
                    from_learning_log_entries=[custom_learning],
                )
            ]

    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator(score=1.5, num_failure_cases=2)
    mutators: list[Mutator] = [CustomMutator()]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
    )

    evolver.evolve_iteration(num_parents=1)

    # Get the mutated organism
    mutated_organism = population.organisms[1][0]
    initial_organism = population.organisms[0][0]

    # Verify custom attributes were preserved
    assert mutated_organism.parent != initial_organism  # Should be custom parent
    assert mutated_organism.from_failure_cases is not None
    assert len(mutated_organism.from_failure_cases) == 1
    assert mutated_organism.from_failure_cases[0].data_point_id == "custom"
    assert mutated_organism.from_learning_log_entries is not None
    assert len(mutated_organism.from_learning_log_entries) == 1
    assert mutated_organism.from_learning_log_entries[0].attempted_change == "custom"


def test_evolver_population_property() -> None:
    """Test the population property accessor."""
    population = create_weighted_population(fixed_midpoint_score=0.5)

    evaluator = MockEvaluator()
    mutators: list[Mutator] = [MockMutator()]

    evolver = Evolver(
        population=population,
        mutators=mutators,
        evaluator=evaluator,
        learning_log_view_type=(EmptyLearningLogView, {}),
    )

    assert evolver.population is population
