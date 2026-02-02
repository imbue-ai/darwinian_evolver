"""Tests for population snapshot/restore functionality."""

import pickle
from uuid import uuid4

import pytest

from darwinian_evolver.population import FixedTreePopulation
from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.test_utils import MockOrganism
from darwinian_evolver.test_utils import add_test_child
from darwinian_evolver.test_utils import create_fixed_tree_population
from darwinian_evolver.test_utils import create_test_organism
from darwinian_evolver.test_utils import create_test_result
from darwinian_evolver.test_utils import create_weighted_population


def test_weighted_sampling_population_snapshot_roundtrip() -> None:
    """Test that WeightedSamplingPopulation can be saved to and restored from a snapshot."""
    # Create initial population with custom parameters
    population = create_weighted_population(
        sharpness=15.0,
        fixed_midpoint_score=None,
        midpoint_score_percentile=80.0,
        novelty_weight=2.0,
    )

    # Add some child organisms to make test more comprehensive
    initial_organism = population.organisms[0][0]
    child1, child1_result = add_test_child(population, initial_organism, score=1.5)
    child2, child2_result = add_test_child(population, initial_organism, score=2.0)
    grandchild, grandchild_result = add_test_child(population, child1, score=2.5)

    # Take snapshot
    snapshot = population.snapshot()

    # Restore from snapshot
    restored_population = WeightedSamplingPopulation.from_snapshot(snapshot)

    # Verify organisms
    assert len(restored_population.organisms) == 4
    assert len(restored_population._organisms_by_id) == 4

    # Verify organism IDs match
    original_ids = {org.id for org, _ in population.organisms}
    restored_ids = {org.id for org, _ in restored_population.organisms}
    assert original_ids == restored_ids

    # Verify scores
    original_scores = [result.score for _, result in population.organisms]
    restored_scores = [result.score for _, result in restored_population.organisms]
    assert original_scores == restored_scores

    # Verify class-specific parameters
    assert restored_population._sharpness == 15.0
    assert restored_population._fixed_midpoint_score is None
    assert restored_population._midpoint_score_percentile == 80.0
    assert restored_population._novelty_weight == 2.0

    # Verify parent-child relationships
    assert len(restored_population._children[initial_organism.id]) == 2
    assert len(restored_population._children[child1.id]) == 1
    assert len(restored_population._children[child2.id]) == 0
    assert len(restored_population._children[grandchild.id]) == 0

    # Verify children IDs match
    assert set(restored_population._children[initial_organism.id]) == {child1.id, child2.id}
    assert restored_population._children[child1.id] == [grandchild.id]

    # Verify get_children works correctly
    restored_children = restored_population.get_children(initial_organism)
    assert len(restored_children) == 2
    restored_child_ids = {org.id for org, _ in restored_children}
    assert restored_child_ids == {child1.id, child2.id}

    # Verify learning log is populated (empty since no organisms have from_change_summary)
    assert len(restored_population.learning_log._entry_for_organism) == 0

    # Verify best organism
    best_org, best_result = restored_population.get_best()
    assert best_result.score == 2.5
    assert best_org.id == grandchild.id


def test_weighted_sampling_population_with_fixed_midpoint() -> None:
    """Test WeightedSamplingPopulation snapshot with fixed_midpoint_score."""
    population = create_weighted_population(
        sharpness=10.0,
        fixed_midpoint_score=1.5,
        novelty_weight=1.0,
    )

    # Add a child organism
    child, child_result = add_test_child(population, score=2.0)

    # Take snapshot and restore
    snapshot = population.snapshot()
    restored_population = WeightedSamplingPopulation.from_snapshot(snapshot)

    # Verify class-specific parameters
    assert restored_population._sharpness == 10.0
    assert restored_population._fixed_midpoint_score == 1.5
    assert restored_population._midpoint_score_percentile is None
    assert restored_population._novelty_weight == 1.0

    # Verify organisms
    assert len(restored_population.organisms) == 2


def test_fixed_tree_population_snapshot_roundtrip() -> None:
    """Test that FixedTreePopulation can be saved to and restored from a snapshot."""
    # Create initial population
    population = create_fixed_tree_population(
        fixed_children_per_generation=[3, 2, 4],
    )

    # Add generation 1 organisms (3 children of initial)
    initial_organism = population.organisms[0][0]
    gen1_organisms = []
    for i in range(3):
        child, child_result = create_test_organism(parent=initial_organism, score=1.0 + i * 0.1)
        population.add(child, child_result)
        gen1_organisms.append((child, child_result))

    # Add generation 2 organisms (2 children of first gen1 organism)
    gen2_organisms = []
    for i in range(2):
        grandchild, grandchild_result = create_test_organism(parent=gen1_organisms[0][0], score=2.0 + i * 0.1)
        population.add(grandchild, grandchild_result)
        gen2_organisms.append((grandchild, grandchild_result))

    # Take snapshot
    snapshot = population.snapshot()

    # Restore from snapshot
    restored_population = FixedTreePopulation.from_snapshot(snapshot)

    # Verify organisms
    assert len(restored_population.organisms) == 6
    assert len(restored_population._organisms_by_id) == 6

    # Verify organism IDs match
    original_ids = {org.id for org, _ in population.organisms}
    restored_ids = {org.id for org, _ in restored_population.organisms}
    assert original_ids == restored_ids

    # Verify class-specific parameters
    assert restored_population._fixed_children_per_generation == [3, 2, 4]

    # Verify parent-child relationships
    assert len(restored_population._children[initial_organism.id]) == 3
    assert len(restored_population._children[gen1_organisms[0][0].id]) == 2
    assert len(restored_population._children[gen1_organisms[1][0].id]) == 0
    assert len(restored_population._children[gen1_organisms[2][0].id]) == 0

    # Verify generation computation
    assert FixedTreePopulation._compute_generation(initial_organism) == 0
    assert all(FixedTreePopulation._compute_generation(org) == 1 for org, _ in gen1_organisms)
    assert all(FixedTreePopulation._compute_generation(org) == 2 for org, _ in gen2_organisms)

    # Verify get_children works correctly
    restored_children = restored_population.get_children(initial_organism)
    assert len(restored_children) == 3

    # Verify best organism
    best_org, best_result = restored_population.get_best()
    assert best_result.score == 2.1  # Last gen2 organism


def test_snapshot_preserves_organism_references() -> None:
    """Test that parent references in organisms are preserved across snapshot/restore."""
    # Create a small family tree
    population = create_weighted_population()
    root = population.organisms[0][0]

    child, child_result = create_test_organism(parent=root, score=1.5)
    population.add(child, child_result)

    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0)
    population.add(grandchild, grandchild_result)

    # Take snapshot and restore
    snapshot = population.snapshot()
    restored = WeightedSamplingPopulation.from_snapshot(snapshot)

    # Find the restored grandchild
    restored_grandchild_org = None
    for org, result in restored.organisms:
        if result.score == 2.0:
            restored_grandchild_org = org
            break

    assert restored_grandchild_org is not None

    # Verify parent chain exists
    assert restored_grandchild_org.parent is not None
    assert restored_grandchild_org.parent.parent is not None
    assert restored_grandchild_org.parent.parent.parent is None

    # Verify parent IDs match
    assert restored_grandchild_org.parent.id == child.id
    assert restored_grandchild_org.parent.parent.id == root.id


def test_fixed_tree_population_requires_children_pattern() -> None:
    """Test that FixedTreePopulation requires fixed_children_per_generation."""
    initial_organism, initial_result = create_test_organism(score=1.0)

    # Should fail without children pattern
    with pytest.raises(ValueError, match="fixed_children_per_generation is required"):
        FixedTreePopulation(
            initial_organism=initial_organism,
            initial_evaluation_result=initial_result,
            fixed_children_per_generation=None,
        )


def test_snapshot_includes_class_name() -> None:
    """Test that snapshots include the class name for proper restoration."""

    # Create WeightedSamplingPopulation
    weighted_pop = create_weighted_population()
    weighted_snapshot = weighted_pop.snapshot()
    weighted_dict = pickle.loads(weighted_snapshot)
    assert weighted_dict["class_name"] == "WeightedSamplingPopulation"

    # Create FixedTreePopulation
    fixed_pop = create_fixed_tree_population(fixed_children_per_generation=[2, 3])
    fixed_snapshot = fixed_pop.snapshot()
    fixed_dict = pickle.loads(fixed_snapshot)
    assert fixed_dict["class_name"] == "FixedTreePopulation"


# ============================================================================
# Tests for basic Population functionality
# ============================================================================


def test_population_initialization_rejects_non_viable_organism() -> None:
    """Test that Population initialization fails with non-viable initial organism."""
    organism = MockOrganism()  # Create organism without parent
    non_viable_result = EvaluationResult(
        score=0.5,
        trainable_failure_cases=[],
        is_viable=False,  # Non-viable
    )

    with pytest.raises(ValueError, match="Initial organism must be viable"):
        WeightedSamplingPopulation(
            initial_organism=organism,
            initial_evaluation_result=non_viable_result,
            sharpness=10.0,
            fixed_midpoint_score=1.0,
            midpoint_score_percentile=None,
        )


def test_population_initialization_rejects_organism_with_parent() -> None:
    """Test that Population initialization fails if initial organism has a parent."""
    parent_org = MockOrganism()
    child_org, child_result = create_test_organism(parent=parent_org, score=1.5)

    with pytest.raises(AssertionError, match="Initial organism must not have a parent"):
        WeightedSamplingPopulation(
            initial_organism=child_org,
            initial_evaluation_result=child_result,
            sharpness=10.0,
            fixed_midpoint_score=1.0,
            midpoint_score_percentile=None,
        )


def test_add_organism() -> None:
    """Test adding organisms to population."""
    population = create_weighted_population()
    assert len(population.organisms) == 1

    # Add a child
    parent = population.organisms[0][0]
    child, child_result = add_test_child(population, parent, score=2.0)

    assert len(population.organisms) == 2
    assert len(population.get_children(parent)) == 1
    assert len(population.get_children(child)) == 0


def test_add_duplicate_organism_fails() -> None:
    """Test that adding the same organism twice fails."""
    population = create_weighted_population()
    parent = population.organisms[0][0]
    child, child_result = create_test_organism(parent=parent, score=2.0)
    population.add(child, child_result)

    # Try to add the same organism again
    with pytest.raises(AssertionError, match="is already in the population"):
        population.add(child, child_result)


def test_add_failed_verification() -> None:
    """Test adding organisms that failed verification."""
    population = create_weighted_population()
    assert len(population._organisms_failed_verification) == 0

    # Add a failed organism
    parent = population.organisms[0][0]
    failed_org, _ = create_test_organism(parent=parent, score=0.5)
    population.add_failed_verification(failed_org)

    assert len(population._organisms_failed_verification) == 1
    assert population._organisms_failed_verification[0] == failed_org


def test_get_best() -> None:
    """Test getting the best organism from population."""
    population = create_weighted_population(score=1.0)
    parent = population.organisms[0][0]

    # Add organisms with various scores
    child1, child1_result = create_test_organism(parent=parent, score=2.0)
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=parent, score=0.5)
    population.add(child2, child2_result)

    child3, child3_result = create_test_organism(parent=parent, score=3.5)
    population.add(child3, child3_result)

    best_org, best_result = population.get_best()
    assert best_result.score == 3.5
    assert best_org.id == child3.id


def test_get_children() -> None:
    """Test getting children of a parent organism."""
    population = create_weighted_population()
    parent = population.organisms[0][0]

    # Add multiple children
    child1, child1_result = create_test_organism(parent=parent, score=1.5)
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=parent, score=2.0)
    population.add(child2, child2_result)

    # Add grandchild
    grandchild, grandchild_result = create_test_organism(parent=child1, score=2.5)
    population.add(grandchild, grandchild_result)

    # Get children of parent
    children = population.get_children(parent)
    assert len(children) == 2
    child_ids = {org.id for org, _ in children}
    assert child_ids == {child1.id, child2.id}

    # Get children of child1
    grandchildren = population.get_children(child1)
    assert len(grandchildren) == 1
    assert grandchildren[0][0].id == grandchild.id

    # Get children of child2 (no children)
    assert len(population.get_children(child2)) == 0


def test_get_score_percentiles_single_organism() -> None:
    """Test score percentiles with a single organism."""
    population = create_weighted_population(score=5.0)

    percentiles = population.get_score_percentiles([0, 25, 50, 75, 100])
    # All percentiles should be the same with single organism
    assert all(percentiles[p] == 5.0 for p in [0, 25, 50, 75, 100])


def test_get_score_percentiles_multiple_organisms() -> None:
    """Test score percentiles calculation with multiple organisms."""
    population = create_weighted_population(score=0.0)
    parent = population.organisms[0][0]

    # Add organisms with scores: 0.0, 1.0, 2.0, 3.0, 4.0
    for i in range(1, 5):
        child, child_result = create_test_organism(parent=parent, score=float(i))
        population.add(child, child_result)

    percentiles = population.get_score_percentiles([0, 25, 50, 75, 100])

    assert percentiles[0] == 0.0  # Min
    assert percentiles[100] == 4.0  # Max
    assert percentiles[50] == 2.0  # Median
    # 25th and 75th percentiles use linear interpolation
    assert percentiles[25] == 1.0
    assert percentiles[75] == 3.0


def test_log_to_json_dict() -> None:
    """Test JSON serialization of population."""
    population = create_weighted_population(score=1.0, change_summary="initial")
    parent = population.organisms[0][0]

    child, child_result = create_test_organism(parent=parent, score=2.0, change_summary="mutation_1")
    population.add(child, child_result)

    # Add a failed verification organism
    failed_org, _ = create_test_organism(parent=parent, score=0.5, change_summary="failed_mutation")
    population.add_failed_verification(failed_org)

    json_dict = population.log_to_json_dict()

    # Check structure
    assert "organisms" in json_dict
    assert "organisms_failed_verification" in json_dict

    # Check organisms
    assert len(json_dict["organisms"]) == 2
    for org_entry in json_dict["organisms"]:
        assert "organism" in org_entry
        assert "evaluation_result" in org_entry
        assert "parent_id" in org_entry["organism"]

    # Check failed verification organisms
    assert len(json_dict["organisms_failed_verification"]) == 1
    assert json_dict["organisms_failed_verification"][0]["from_change_summary"] == "failed_mutation"


def test_learning_log_property() -> None:
    """Test that learning_log property returns the learning log."""
    population = create_weighted_population()
    learning_log = population.learning_log

    assert learning_log is not None
    assert len(learning_log._entry_for_organism) == 0  # No change summaries yet


def test_learning_log_with_change_summaries() -> None:
    """Test that learning log is populated when organisms have change summaries."""
    # The initial organism needs a change summary, but it won't be added to the log since it has no parent
    population = create_weighted_population(score=1.0, change_summary="initial_change")
    parent = population.organisms[0][0]

    # Add children with change summaries
    child1, child1_result = create_test_organism(parent=parent, score=1.5, change_summary="change_1")
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=parent, score=2.0, change_summary="change_2")
    population.add(child2, child2_result)

    learning_log = population.learning_log
    # The initial organism with change_summary="initial_change" has no parent, so it IS added to the log
    # Children with change summaries are also added
    # Total: 1 (initial) + 2 (children) = 3
    assert len(learning_log._entry_for_organism) == 3


# ============================================================================
# Tests for WeightedSamplingPopulation
# ============================================================================


@pytest.mark.parametrize(
    "params,expected_error",
    [
        (
            {"sharpness": 0.0, "fixed_midpoint_score": 1.0, "midpoint_score_percentile": None},
            "Sharpness must be positive",
        ),
        (
            {"sharpness": 10.0, "fixed_midpoint_score": 1.0, "midpoint_score_percentile": 75.0},
            "Exactly one of fixed_midpoint_score or midpoint_score_percentile",
        ),
        (
            {"sharpness": 10.0, "fixed_midpoint_score": None, "midpoint_score_percentile": None},
            "Exactly one of fixed_midpoint_score or midpoint_score_percentile",
        ),
        (
            {
                "sharpness": 10.0,
                "fixed_midpoint_score": 1.0,
                "midpoint_score_percentile": None,
                "novelty_weight": -1.0,
            },
            "Novelty weight must be non-negative",
        ),
    ],
)
def test_weighted_sampling_population_initialization_validation(params: dict, expected_error: str) -> None:
    """Test validation of WeightedSamplingPopulation initialization parameters."""
    organism = MockOrganism()
    result = create_test_result()
    with pytest.raises(AssertionError, match=expected_error):
        WeightedSamplingPopulation(initial_organism=organism, initial_evaluation_result=result, **params)


def test_weighted_sampling_set_sharpness() -> None:
    """Test setting sharpness parameter."""
    population = create_weighted_population(sharpness=10.0)
    assert population._sharpness == 10.0

    population.set_sharpness(20.0)
    assert population._sharpness == 20.0

    with pytest.raises(AssertionError, match="Sharpness must be positive"):
        population.set_sharpness(0.0)


def test_weighted_sampling_set_fixed_midpoint_score() -> None:
    """Test setting fixed midpoint score."""
    population = create_weighted_population(fixed_midpoint_score=1.0, midpoint_score_percentile=None)
    assert population._fixed_midpoint_score == 1.0
    assert population._midpoint_score_percentile is None

    population.set_fixed_midpoint_score(2.0)
    assert population._fixed_midpoint_score == 2.0
    assert population._midpoint_score_percentile is None


def test_weighted_sampling_set_midpoint_score_percentile() -> None:
    """Test setting midpoint score percentile."""
    population = create_weighted_population(fixed_midpoint_score=1.0, midpoint_score_percentile=None)

    population.set_midpoint_score_percentile(80.0)
    assert population._midpoint_score_percentile == 80.0
    assert population._fixed_midpoint_score is None

    # Test validation
    with pytest.raises(ValueError, match="midpoint_score_percentile must be between 0 and 100"):
        population.set_midpoint_score_percentile(150.0)

    with pytest.raises(ValueError, match="midpoint_score_percentile must be between 0 and 100"):
        population.set_midpoint_score_percentile(-10.0)


def test_weighted_sampling_set_novelty_weight() -> None:
    """Test setting novelty weight."""
    population = create_weighted_population(novelty_weight=1.0)
    assert population._novelty_weight == 1.0

    population.set_novelty_weight(2.5)
    assert population._novelty_weight == 2.5

    with pytest.raises(AssertionError, match="Novelty weight must be non-negative"):
        population.set_novelty_weight(-1.0)


def test_weighted_sampling_sample_parents_with_replacement() -> None:
    """Test sampling parents with replacement."""
    population = create_weighted_population(score=1.0)
    parent = population.organisms[0][0]

    # Add more organisms with varying scores and failure cases
    for i in range(1, 5):
        add_test_child(population, parent, score=float(i))

    # Sample with replacement
    samples = population.sample_parents(k=10, replace=True)
    assert len(samples) == 10

    # All samples should be viable with failure cases
    for org, result in samples:
        assert result.is_viable
        assert len(result.trainable_failure_cases) > 0


def test_weighted_sampling_sample_parents_without_replacement() -> None:
    """Test sampling parents without replacement."""
    population = create_weighted_population(score=1.0)
    parent = population.organisms[0][0]

    # Add 4 more organisms
    children = []
    for i in range(1, 5):
        child = MockOrganism(parent=parent)
        child_result = EvaluationResult(
            score=float(i),
            trainable_failure_cases=[
                EvaluationFailureCase(data_point_id=f"fail_{uuid4().hex[:8]}", failure_type="test")
            ],
            is_viable=True,
        )
        population.add(child, child_result)
        children.append((child, child_result))

    # Sample without replacement - should get unique organisms
    samples = population.sample_parents(k=3, replace=False)
    assert len(samples) == 3

    # Check that all samples are unique
    sample_ids = [org.id for org, _ in samples]
    assert len(set(sample_ids)) == 3


def test_weighted_sampling_sample_parents_without_replacement_fails_when_k_too_large() -> None:
    """Test that sampling without replacement fails when k > eligible organisms."""
    population = create_weighted_population(score=1.0)
    # Only 1 organism in population

    with pytest.raises(ValueError, match="Cannot sample .* parents without replacement"):
        population.sample_parents(k=5, replace=False)


def test_weighted_sampling_no_eligible_organisms() -> None:
    """Test that sampling fails when no organisms are eligible."""
    # Create organism with no trainable failure cases
    organism, _ = create_test_organism(score=1.0)
    no_failure_result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[],  # No failure cases
        is_viable=True,
    )

    population = create_weighted_population()
    # Replace the initial organism with one that has no failure cases
    population._organisms = [(organism, no_failure_result)]
    population._organisms_by_id = {organism.id: (organism, no_failure_result)}

    with pytest.raises(RuntimeError, match="No eligible organisms for parent selection"):
        population.sample_parents(k=1)


def test_weighted_sampling_compute_weights_with_fixed_midpoint() -> None:
    """Test weight computation with fixed midpoint score."""
    population = create_weighted_population(
        score=1.0,
        sharpness=10.0,
        fixed_midpoint_score=2.0,
        novelty_weight=1.0,
    )
    parent = population.organisms[0][0]

    # Add organisms with scores above and below midpoint
    child1 = MockOrganism(parent=parent)
    child1_result = EvaluationResult(
        score=3.0,  # Above midpoint
        trainable_failure_cases=[EvaluationFailureCase(data_point_id=f"fail_{uuid4().hex[:8]}", failure_type="test")],
        is_viable=True,
    )
    population.add(child1, child1_result)

    child2 = MockOrganism(parent=parent)
    child2_result = EvaluationResult(
        score=1.0,  # Below midpoint
        trainable_failure_cases=[EvaluationFailureCase(data_point_id=f"fail_{uuid4().hex[:8]}", failure_type="test")],
        is_viable=True,
    )
    population.add(child2, child2_result)

    # Compute weights
    eligible_organisms = [
        (org, result)
        for org, result in population.organisms
        if result.is_viable and len(result.trainable_failure_cases) > 0
    ]
    weights = population._compute_weights(eligible_organisms, population._novelty_weight)

    # Organism with score above midpoint should have higher weight
    # Find indices
    high_score_idx = next(i for i, (org, _) in enumerate(eligible_organisms) if org.id == child1.id)
    low_score_idx = next(i for i, (org, _) in enumerate(eligible_organisms) if org.id == child2.id)

    assert weights[high_score_idx] > weights[low_score_idx]


def test_weighted_sampling_compute_weights_with_percentile_midpoint() -> None:
    """Test weight computation with percentile-based midpoint score."""
    population = create_weighted_population(
        score=1.0,
        sharpness=10.0,
        fixed_midpoint_score=None,
        midpoint_score_percentile=50.0,
        novelty_weight=1.0,
    )
    parent = population.organisms[0][0]

    # Add organisms with various scores
    for score in [2.0, 3.0, 4.0, 5.0]:
        child = MockOrganism(parent=parent)
        child_result = EvaluationResult(
            score=score,
            trainable_failure_cases=[
                EvaluationFailureCase(data_point_id=f"fail_{uuid4().hex[:8]}", failure_type="test")
            ],
            is_viable=True,
        )
        population.add(child, child_result)

    # Compute weights - midpoint should be computed dynamically
    eligible_organisms = [
        (org, result)
        for org, result in population.organisms
        if result.is_viable and len(result.trainable_failure_cases) > 0
    ]
    weights = population._compute_weights(eligible_organisms, population._novelty_weight)

    # All weights should be non-negative
    assert all(w >= 0 for w in weights)


def test_weighted_sampling_novelty_bonus() -> None:
    """Test that novelty bonus decreases as more children are added."""
    population = create_weighted_population(
        score=2.0,
        sharpness=10.0,
        fixed_midpoint_score=2.0,
        novelty_weight=2.0,
    )
    parent = population.organisms[0][0]

    # Compute initial novelty bonus (no children)
    initial_bonus = population._compute_novelty_bonus(parent, population._novelty_weight)

    # Add children to parent
    for i in range(3):
        child = MockOrganism(parent=parent)
        child_result = EvaluationResult(
            score=2.0,
            trainable_failure_cases=[
                EvaluationFailureCase(data_point_id=f"fail_{uuid4().hex[:8]}", failure_type="test")
            ],
            is_viable=True,
        )
        population.add(child, child_result)

    # Compute novelty bonus after adding children
    final_bonus = population._compute_novelty_bonus(parent, population._novelty_weight)

    # Novelty bonus should decrease
    assert final_bonus < initial_bonus


# ============================================================================
# Tests for FixedTreePopulation
# ============================================================================


def test_fixed_tree_population_requires_non_empty_children_pattern() -> None:
    """Test that FixedTreePopulation requires non-empty children pattern."""
    organism, result = create_test_organism(score=1.0)

    with pytest.raises(AssertionError, match="fixed_children_per_generation must be non-empty"):
        FixedTreePopulation(
            initial_organism=organism,
            initial_evaluation_result=result,
            fixed_children_per_generation=[],
        )


def test_fixed_tree_population_requires_positive_child_counts() -> None:
    """Test that FixedTreePopulation requires positive child counts."""
    organism, result = create_test_organism(score=1.0)

    with pytest.raises(AssertionError, match="All child counts must be positive"):
        FixedTreePopulation(
            initial_organism=organism,
            initial_evaluation_result=result,
            fixed_children_per_generation=[3, 0, 2],
        )


def test_fixed_tree_sample_parents_requires_iteration() -> None:
    """Test that FixedTreePopulation.sample_parents requires iteration parameter."""
    population = create_fixed_tree_population(fixed_children_per_generation=[2, 3])

    with pytest.raises(ValueError, match="FixedTreePopulation requires iteration parameter"):
        population.sample_parents(k=5, iteration=None)


def test_fixed_tree_sample_parents_returns_correct_pattern() -> None:
    """Test that FixedTreePopulation returns parents according to the pattern."""
    population = create_fixed_tree_population(fixed_children_per_generation=[3, 2, 4])

    # Iteration 0: each frontier organism should appear 3 times
    parents_iter0 = population.sample_parents(k=0, iteration=0)
    assert len(parents_iter0) == 3  # 1 organism * 3 children

    # Add generation 1
    initial_organism = population.organisms[0][0]
    for i in range(3):
        child, child_result = create_test_organism(parent=initial_organism, score=1.0 + i * 0.1)
        population.add(child, child_result)

    # Iteration 1: each frontier organism should appear 2 times
    parents_iter1 = population.sample_parents(k=0, iteration=1)
    assert len(parents_iter1) == 6  # 3 organisms * 2 children

    # Iteration 2: each frontier organism should appear 4 times
    parents_iter2 = population.sample_parents(k=0, iteration=2)
    assert len(parents_iter2) == 12  # 3 organisms * 4 children

    # Iteration 3: pattern [3, 2, 4] wraps around (3 % 3 = 0), so uses first pattern (3 children)
    parents_iter3 = population.sample_parents(k=0, iteration=3)
    assert len(parents_iter3) == 9  # 3 organisms * 3 children (index 0 of pattern)


def test_fixed_tree_compute_generation() -> None:
    """Test generation computation."""
    population = create_fixed_tree_population(fixed_children_per_generation=[2])
    initial_organism = population.organisms[0][0]

    assert FixedTreePopulation._compute_generation(initial_organism) == 0

    # Add child
    child, child_result = create_test_organism(parent=initial_organism, score=1.5)
    population.add(child, child_result)
    assert FixedTreePopulation._compute_generation(child) == 1

    # Add grandchild
    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0)
    population.add(grandchild, grandchild_result)
    assert FixedTreePopulation._compute_generation(grandchild) == 2


def test_fixed_tree_get_current_generation_frontier() -> None:
    """Test getting the current generation frontier."""
    population = create_fixed_tree_population(fixed_children_per_generation=[2])
    initial_organism = population.organisms[0][0]

    # Initially, frontier is just the initial organism
    frontier = population._get_current_generation_frontier()
    assert len(frontier) == 1
    assert frontier[0][0].id == initial_organism.id

    # Add generation 1
    gen1_children = []
    for i in range(3):
        child, child_result = create_test_organism(parent=initial_organism, score=1.0 + i * 0.1)
        population.add(child, child_result)
        gen1_children.append(child)

    # Frontier should now be generation 1
    frontier = population._get_current_generation_frontier()
    assert len(frontier) == 3
    frontier_ids = {org.id for org, _ in frontier}
    assert frontier_ids == {child.id for child in gen1_children}

    # Add one grandchild
    grandchild, grandchild_result = create_test_organism(parent=gen1_children[0], score=2.0)
    population.add(grandchild, grandchild_result)

    # Frontier should now be generation 2 (just the grandchild)
    frontier = population._get_current_generation_frontier()
    assert len(frontier) == 1
    assert frontier[0][0].id == grandchild.id


def test_fixed_tree_empty_population_raises_error() -> None:
    """Test that sample_parents fails on empty frontier (shouldn't happen in practice)."""
    population = create_fixed_tree_population(fixed_children_per_generation=[2])
    # Artificially empty the population
    population._organisms = []

    with pytest.raises(RuntimeError, match="No organisms in current generation frontier"):
        population.sample_parents(k=1, iteration=0)
