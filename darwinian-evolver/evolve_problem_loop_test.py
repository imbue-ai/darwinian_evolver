"""Tests for EvolveProblemLoop snapshot loading functionality."""

import pickle

import pytest

from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.learning_log_view import AncestorLearningLogView
from darwinian_evolver.learning_log_view import EmptyLearningLogView
from darwinian_evolver.population import FixedTreePopulation
from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.test_utils import MockOrganism
from darwinian_evolver.test_utils import add_test_child
from darwinian_evolver.test_utils import create_mock_problem
from darwinian_evolver.test_utils import create_test_loop


def test_weighted_sampling_population_snapshot_loading() -> None:
    """Test that EvolveProblemLoop can load a WeightedSamplingPopulation snapshot."""
    # Create initial loop with weighted sampling
    loop = create_test_loop(
        num_parents_per_iteration=2,
        sharpness=15.0,
        midpoint_score_percentile=80.0,
        novelty_weight=2.0,
    )

    # Initialize the evolver (evaluates initial organism)
    loop._initialize_evolver()

    # Manually add a few organisms to the population
    population = loop._guaranteed_evolver.population
    add_test_child(population, score=1.5)
    add_test_child(population, score=2.0)

    # Set iteration counter
    loop._current_iteration = 5

    # Create a snapshot using the loop's own method
    snapshot = loop._snapshot()

    # Create a new loop from the snapshot
    resumed_loop = create_test_loop(
        num_parents_per_iteration=2,
        snapshot_to_resume_from=snapshot,
    )

    # Verify the loop state is restored
    assert resumed_loop._current_iteration == 5
    assert resumed_loop._evolver is not None
    assert len(resumed_loop._guaranteed_evolver.population.organisms) == 3

    # Verify population parameters are restored
    restored_population = resumed_loop._guaranteed_evolver.population
    assert isinstance(restored_population, WeightedSamplingPopulation)
    assert restored_population._sharpness == 15.0
    assert restored_population._midpoint_score_percentile == 80.0
    assert restored_population._novelty_weight == 2.0

    # Verify snapshot contains class name
    snapshot_dict = pickle.loads(snapshot)
    population_snapshot_dict = pickle.loads(snapshot_dict["population_snapshot"])
    assert population_snapshot_dict["class_name"] == "WeightedSamplingPopulation"


def test_fixed_tree_population_snapshot_loading() -> None:
    """Test that EvolveProblemLoop can load a FixedTreePopulation snapshot."""
    # Create initial loop with fixed tree mode
    loop = create_test_loop(
        num_parents_per_iteration=2,
        fixed_children_per_generation=[3, 2],
    )

    # Initialize the evolver (evaluates initial organism)
    loop._initialize_evolver()

    # Manually add a few organisms to the population
    population = loop._guaranteed_evolver.population
    add_test_child(population, score=1.5)
    add_test_child(population, score=2.0)

    # Set iteration counter
    loop._current_iteration = 3

    # Create a snapshot using the loop's own method
    snapshot = loop._snapshot()

    # Create a new loop from the snapshot
    resumed_loop = create_test_loop(
        num_parents_per_iteration=2,
        snapshot_to_resume_from=snapshot,
    )

    # Verify the loop state is restored
    assert resumed_loop._current_iteration == 3
    assert resumed_loop._evolver is not None
    assert len(resumed_loop._guaranteed_evolver.population.organisms) == 3

    # Verify population parameters are restored
    restored_population = resumed_loop._guaranteed_evolver.population
    assert isinstance(restored_population, FixedTreePopulation)
    assert restored_population._fixed_children_per_generation == [3, 2]

    # Verify snapshot contains class name
    snapshot_dict = pickle.loads(snapshot)
    population_snapshot_dict = pickle.loads(snapshot_dict["population_snapshot"])
    assert population_snapshot_dict["class_name"] == "FixedTreePopulation"


def test_evolve_problem_loop_initialization_weighted_sampling() -> None:
    """Test that EvolveProblemLoop initializes correctly with weighted sampling parameters."""
    loop = create_test_loop(
        mutator_concurrency=5,
        evaluator_concurrency=3,
        num_parents_per_iteration=4,
        sharpness=12.0,
        midpoint_score_percentile=70.0,
        novelty_weight=1.5,
        batch_size=2,
        should_verify_mutations=True,
    )

    # Verify parameters are stored
    assert loop._mutator_concurrency == 5
    assert loop._evaluator_concurrency == 3
    assert loop._num_parents_per_iteration == 4
    assert loop._sharpness == 12.0
    assert loop._midpoint_score_percentile == 70.0
    assert loop._novelty_weight == 1.5
    assert loop._batch_size == 2
    assert loop._should_verify_mutations is True
    assert loop._current_iteration == 0
    assert loop._evolver is None


def test_evolve_problem_loop_initialization_fixed_tree() -> None:
    """Test that EvolveProblemLoop initializes correctly with fixed tree parameters."""
    loop = create_test_loop(fixed_children_per_generation=[5, 3, 2])

    assert loop._fixed_children_per_generation == [5, 3, 2]
    assert loop._evolver is None


def test_evolve_problem_loop_initialization_conflicting_midpoint() -> None:
    """Test that initialization fails with both fixed and percentile midpoint scores."""
    problem = create_mock_problem()

    with pytest.raises(AssertionError, match="Cannot specify both"):
        EvolveProblemLoop(
            problem=problem,
            learning_log_view_type=(EmptyLearningLogView, {}),
            fixed_midpoint_score=1.5,
            midpoint_score_percentile=75.0,
        )


def test_run_single_iteration() -> None:
    """Test running a single evolution iteration."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))

    # Should get 2 snapshots: initial + 1 iteration
    assert len(snapshots) == 2

    # First snapshot is after evaluating initial organism
    assert snapshots[0].iteration == 0
    assert snapshots[0].population_size == 1

    # Second snapshot is after first iteration
    assert snapshots[1].iteration == 1
    assert snapshots[1].population_size > 1


def test_run_multiple_iterations() -> None:
    """Test running multiple evolution iterations."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=3))

    # Should get 4 snapshots: initial + 3 iterations
    assert len(snapshots) == 4
    assert snapshots[0].iteration == 0
    assert snapshots[1].iteration == 1
    assert snapshots[2].iteration == 2
    assert snapshots[3].iteration == 3


def test_run_zero_iterations() -> None:
    """Test that running zero iterations raises an error."""
    loop = create_test_loop()

    with pytest.raises(ValueError, match="Number of iterations must be positive"):
        list(loop.run(num_iterations=0))


def test_run_negative_iterations() -> None:
    """Test that running negative iterations raises an error."""
    loop = create_test_loop()

    with pytest.raises(ValueError, match="Number of iterations must be positive"):
        list(loop.run(num_iterations=-1))


def test_iteration_snapshot_contents() -> None:
    """Test that IterationSnapshot contains all expected fields."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))
    snapshot = snapshots[0]

    # Verify all fields are present
    assert isinstance(snapshot.iteration, int)
    assert isinstance(snapshot.population_size, int)
    assert isinstance(snapshot.snapshot, bytes)
    assert isinstance(snapshot.population_json_log, dict)
    assert isinstance(snapshot.best_organism_result, tuple)
    assert isinstance(snapshot.score_percentiles, dict)
    assert hasattr(snapshot.evolver_stats, "num_evaluate_calls")


def test_snapshot_after_resume() -> None:
    """Test that snapshots can be taken and resumed multiple times."""
    problem = create_mock_problem()

    # Create initial loop and run 2 iterations
    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
    )

    snapshots1 = list(loop1.run(num_iterations=2))
    snapshot_after_2 = snapshots1[-1].snapshot

    # Resume from snapshot and run 1 more iteration
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        snapshot_to_resume_from=snapshot_after_2,
    )

    snapshots2 = list(loop2.run(num_iterations=1))

    # Should start from iteration 2 and produce iteration 3
    assert snapshots2[0].iteration == 3


def test_snapshot_invalid_format() -> None:
    """Test that loading an invalid snapshot raises an error."""
    problem = create_mock_problem()

    # Try to load a snapshot that's not a pickled dict
    invalid_snapshot = pickle.dumps("not a dict")

    with pytest.raises(ValueError, match="Snapshot must be a pickled dictionary"):
        EvolveProblemLoop(
            problem=problem,
            learning_log_view_type=(EmptyLearningLogView, {}),
            snapshot_to_resume_from=invalid_snapshot,
        )


def test_initialize_evolver_default_parameters() -> None:
    """Test that evolver initialization uses default parameters when not specified."""
    loop = create_test_loop(num_parents_per_iteration=1)

    loop._initialize_evolver()

    # Verify default parameters are used
    population = loop._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._sharpness == EvolveProblemLoop._DEFAULT_SHARPNESS
    assert population._midpoint_score_percentile == EvolveProblemLoop._DEFAULT_MIDPOINT_SCORE_PERCENTILE
    assert population._novelty_weight == EvolveProblemLoop._DEFAULT_NOVELTY_WEIGHT


def test_initialize_evolver_custom_parameters() -> None:
    """Test that evolver initialization uses custom parameters when specified."""
    loop = create_test_loop(
        num_parents_per_iteration=1, sharpness=15.0, midpoint_score_percentile=80.0, novelty_weight=2.0
    )

    loop._initialize_evolver()

    population = loop._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._sharpness == 15.0
    assert population._midpoint_score_percentile == 80.0
    assert population._novelty_weight == 2.0


def test_initialize_evolver_fixed_midpoint_score() -> None:
    """Test evolver initialization with fixed midpoint score."""
    loop = create_test_loop(num_parents_per_iteration=1, fixed_midpoint_score=1.5)

    loop._initialize_evolver()

    population = loop._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._fixed_midpoint_score == 1.5
    assert population._midpoint_score_percentile is None


def test_snapshot_parameter_override() -> None:
    """Test that explicit parameters override snapshot parameters on resume."""
    problem = create_mock_problem()

    # Create initial loop with certain parameters
    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        sharpness=10.0,
        midpoint_score_percentile=75.0,
    )

    loop1._initialize_evolver()
    snapshot = loop1._snapshot()

    # Resume with different parameters
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        sharpness=20.0,
        midpoint_score_percentile=90.0,
        novelty_weight=3.0,
        snapshot_to_resume_from=snapshot,
    )

    # Verify new parameters override snapshot parameters
    population = loop2._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._sharpness == 20.0
    assert population._midpoint_score_percentile == 90.0
    assert population._novelty_weight == 3.0


def test_fixed_tree_parameter_override() -> None:
    """Test that fixed_children_per_generation can be overridden on resume."""
    problem = create_mock_problem()

    # Create initial loop with fixed tree
    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        fixed_children_per_generation=[3, 2],
    )

    loop1._initialize_evolver()
    snapshot = loop1._snapshot()

    # Resume with different pattern
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        fixed_children_per_generation=[5, 4, 3],
        snapshot_to_resume_from=snapshot,
    )

    # Verify new pattern overrides snapshot pattern
    population = loop2._guaranteed_evolver.population
    assert isinstance(population, FixedTreePopulation)
    assert population._fixed_children_per_generation == [5, 4, 3]


def test_learning_log_view_type_applied() -> None:
    """Test that the learning_log_view_type is properly applied to the evolver."""
    problem = create_mock_problem()

    loop = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(AncestorLearningLogView, {"max_depth": 3}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
    )

    loop._initialize_evolver()

    # Verify the learning log view is of the correct type
    assert isinstance(loop._guaranteed_evolver._learning_log_view, AncestorLearningLogView)


def test_mutator_context_set() -> None:
    """Test that mutators receive the population context during initialization."""
    problem = create_mock_problem()
    loop = create_test_loop(problem=problem, num_parents_per_iteration=1)

    loop._initialize_evolver()

    # Verify that mutators have context set
    for mutator in problem.mutators:
        assert mutator._context is not None
        assert mutator._context.population is loop._guaranteed_evolver.population


def test_run_increments_iteration_counter() -> None:
    """Test that the iteration counter increments correctly."""
    loop = create_test_loop(num_parents_per_iteration=1)

    # Initial counter should be 0
    assert loop._current_iteration == 0

    # Run 3 iterations
    list(loop.run(num_iterations=3))

    # Counter should be 3 after 3 iterations
    assert loop._current_iteration == 3


def test_run_without_initialization() -> None:
    """Test that run() initializes evolver if not already initialized."""
    loop = create_test_loop(num_parents_per_iteration=1)

    # Evolver should be None before first run
    assert loop._evolver is None

    # Run should initialize it
    snapshots = list(loop.run(num_iterations=1))

    # Evolver should be initialized after run
    assert loop._evolver is not None

    # Should get initial snapshot + 1 iteration
    assert len(snapshots) == 2


def test_population_json_log_in_snapshot() -> None:
    """Test that population JSON log is included in iteration snapshots."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))
    json_log = snapshots[0].population_json_log

    # Verify JSON log has expected structure
    assert "organisms" in json_log
    assert isinstance(json_log["organisms"], list)
    assert len(json_log["organisms"]) > 0


def test_score_percentiles_in_snapshot() -> None:
    """Test that score percentiles are computed and included in snapshots."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))
    percentiles = snapshots[1].score_percentiles

    # Verify percentiles are present
    assert isinstance(percentiles, dict)
    assert len(percentiles) > 0
    assert 50.0 in percentiles  # Median should be present


def test_best_organism_in_snapshot() -> None:
    """Test that the best organism is correctly identified in snapshots."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))
    best_organism, best_result = snapshots[0].best_organism_result

    # Verify best organism is returned
    assert isinstance(best_organism, MockOrganism)
    assert isinstance(best_result, EvaluationResult)
    assert isinstance(best_result.score, float)


def test_use_process_pool_executors() -> None:
    """Test that use_process_pool_executors parameter is passed to evolver."""
    loop = create_test_loop(num_parents_per_iteration=1, use_process_pool_executors=True)

    loop._initialize_evolver()

    # Verify parameter is passed to evolver
    assert loop._guaranteed_evolver._use_process_pool_executors is True


def test_load_snapshot_with_unrecognized_population_class() -> None:
    """Test that loading a snapshot with an unrecognized population class raises an error."""
    problem = create_mock_problem()

    # Create a snapshot with an invalid class name
    loop = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
    )
    loop._initialize_evolver()
    snapshot = loop._snapshot()

    # Manipulate the snapshot to have an invalid class name
    snapshot_dict = pickle.loads(snapshot)
    population_snapshot_dict = pickle.loads(snapshot_dict["population_snapshot"])
    population_snapshot_dict["class_name"] = "InvalidPopulationClass"
    snapshot_dict["population_snapshot"] = pickle.dumps(population_snapshot_dict)
    invalid_snapshot = pickle.dumps(snapshot_dict)

    # Try to load the invalid snapshot
    with pytest.raises(ValueError, match="Unrecognized Population class"):
        EvolveProblemLoop(
            problem=problem,
            learning_log_view_type=(EmptyLearningLogView, {}),
            snapshot_to_resume_from=invalid_snapshot,
        )


def test_multiple_consecutive_runs() -> None:
    """Test that run() can be called multiple times consecutively."""
    loop = create_test_loop(num_parents_per_iteration=1)

    # First run
    snapshots1 = list(loop.run(num_iterations=2))
    # Should get: initial (iteration 0) + 2 iterations (1, 2)
    assert len(snapshots1) == 3
    assert snapshots1[0].iteration == 0
    assert snapshots1[1].iteration == 1
    assert snapshots1[2].iteration == 2

    # Second run (should continue from where we left off)
    snapshots2 = list(loop.run(num_iterations=2))
    # Should get: 2 more iterations (3, 4)
    assert len(snapshots2) == 2
    assert snapshots2[0].iteration == 3
    assert snapshots2[1].iteration == 4

    # Verify final iteration counter
    assert loop._current_iteration == 4


def test_batch_size_parameter() -> None:
    """Test that batch_size parameter is properly stored and passed to evolver."""
    loop = create_test_loop(num_parents_per_iteration=1, batch_size=5)

    loop._initialize_evolver()

    # Verify batch_size is passed to evolver
    assert loop._guaranteed_evolver._batch_size == 5


def test_should_verify_mutations_parameter() -> None:
    """Test that should_verify_mutations parameter is properly stored and passed to evolver."""
    loop = create_test_loop(num_parents_per_iteration=1, should_verify_mutations=True)

    loop._initialize_evolver()

    # Verify should_verify_mutations is passed to evolver
    assert loop._guaranteed_evolver._should_verify_mutations is True


def test_evolver_stats_in_initial_snapshot() -> None:
    """Test that evolver stats in initial snapshot have correct evaluate count."""
    loop = create_test_loop(num_parents_per_iteration=1)

    snapshots = list(loop.run(num_iterations=1))

    # Initial snapshot should have num_evaluate_calls == 1 (for initial organism)
    initial_snapshot = snapshots[0]
    assert initial_snapshot.evolver_stats.num_evaluate_calls == 1


def test_fixed_midpoint_score_with_no_percentile() -> None:
    """Test that using fixed_midpoint_score results in None for midpoint_score_percentile."""
    loop = create_test_loop(num_parents_per_iteration=1, fixed_midpoint_score=2.5)

    loop._initialize_evolver()

    population = loop._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._fixed_midpoint_score == 2.5
    # When fixed_midpoint_score is set, percentile should be None
    assert population._midpoint_score_percentile is None


def test_resume_with_fixed_midpoint_score_override() -> None:
    """Test that resuming with fixed_midpoint_score overrides the snapshot value."""
    problem = create_mock_problem()

    # Create initial loop with percentile
    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        midpoint_score_percentile=80.0,
    )

    loop1._initialize_evolver()
    snapshot = loop1._snapshot()

    # Resume with fixed_midpoint_score (should override)
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        fixed_midpoint_score=3.0,
        snapshot_to_resume_from=snapshot,
    )

    # Verify fixed_midpoint_score is set
    population = loop2._guaranteed_evolver.population
    assert isinstance(population, WeightedSamplingPopulation)
    assert population._fixed_midpoint_score == 3.0


def test_snapshot_serialization_round_trip() -> None:
    """Test that snapshot can be serialized and deserialized correctly."""
    problem = create_mock_problem()

    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
    )

    # Run some iterations
    snapshots = list(loop1.run(num_iterations=2))
    iteration_snapshot = snapshots[-1]

    # Verify snapshot can be unpickled and has expected structure
    snapshot_bytes = iteration_snapshot.snapshot
    snapshot_dict = pickle.loads(snapshot_bytes)

    assert "population_snapshot" in snapshot_dict
    assert "current_iteration" in snapshot_dict
    assert isinstance(snapshot_dict["population_snapshot"], bytes)
    assert snapshot_dict["current_iteration"] == 2

    # Verify it can be loaded
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        snapshot_to_resume_from=snapshot_bytes,
    )

    assert loop2._current_iteration == 2
    assert loop2._evolver is not None


def test_mutator_context_updated_on_resume() -> None:
    """Test that mutators receive updated context when resuming from snapshot."""
    problem = create_mock_problem()

    # Create initial loop
    loop1 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
    )

    loop1._initialize_evolver()
    snapshot = loop1._snapshot()

    # Resume from snapshot (uses same problem, so same mutator instances)
    loop2 = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(EmptyLearningLogView, {}),
        mutator_concurrency=1,
        evaluator_concurrency=1,
        num_parents_per_iteration=1,
        snapshot_to_resume_from=snapshot,
    )

    # Verify mutators have context pointing to the new population
    for mutator in loop2._guaranteed_evolver._mutators:
        assert mutator._context is not None
        assert mutator._context.population is loop2._guaranteed_evolver.population


def test_population_size_increases_over_iterations() -> None:
    """Test that population size generally increases over iterations."""
    loop = create_test_loop(num_parents_per_iteration=2)

    snapshots = list(loop.run(num_iterations=3))

    # Population should grow over iterations
    # Initial snapshot has 1 organism
    assert snapshots[0].population_size == 1

    # Subsequent snapshots should have more organisms
    for i in range(1, len(snapshots)):
        assert snapshots[i].population_size > snapshots[i - 1].population_size
