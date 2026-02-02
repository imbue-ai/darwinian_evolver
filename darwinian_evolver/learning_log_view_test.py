"""Tests for LearningLogView classes."""

from darwinian_evolver.learning_log_view import AncestorLearningLogView
from darwinian_evolver.learning_log_view import EmptyLearningLogView
from darwinian_evolver.learning_log_view import NeighborhoodLearningLogView
from darwinian_evolver.test_utils import add_test_child
from darwinian_evolver.test_utils import create_test_organism
from darwinian_evolver.test_utils import create_weighted_population

# ============================================================================
# EmptyLearningLogView Tests
# ============================================================================


def test_empty_learning_log_view_returns_empty() -> None:
    """Test that EmptyLearningLogView always returns empty list."""
    population = create_weighted_population(change_summary="initial")
    initial_organism = population.organisms[0][0]

    view = EmptyLearningLogView(population)
    entries = view.get_entries_for_organism(initial_organism)

    assert entries == []


def test_empty_learning_log_view_with_children() -> None:
    """Test that EmptyLearningLogView returns empty even when organism has learning log."""
    population = create_weighted_population(change_summary="initial")

    initial_organism = population.organisms[0][0]

    # Add child with learning log entry
    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    view = EmptyLearningLogView(population)
    entries = view.get_entries_for_organism(child)

    # Should still return empty
    assert entries == []


# ============================================================================
# AncestorLearningLogView Tests
# ============================================================================


def test_ancestor_learning_log_view_no_ancestors() -> None:
    """Test AncestorLearningLogView with root organism (no ancestors)."""
    population = create_weighted_population(score=1.0, change_summary="initial")
    initial_organism = population.organisms[0][0]

    view = AncestorLearningLogView(population)
    entries = view.get_entries_for_organism(initial_organism)

    # Root organism includes itself (since it has a change_summary)
    assert len(entries) == 1
    assert entries[0].attempted_change == "initial"


def test_ancestor_learning_log_view_with_parent() -> None:
    """Test AncestorLearningLogView with one level of ancestry."""
    population = create_weighted_population(score=1.0, change_summary="initial")

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    view = AncestorLearningLogView(population)
    entries = view.get_entries_for_organism(child)

    # Child should see its own entry and parent's entry
    assert len(entries) == 2
    change_summaries = [entry.attempted_change for entry in entries]
    assert "child change" in change_summaries
    assert "initial" in change_summaries


def test_ancestor_learning_log_view_with_grandparent() -> None:
    """Test AncestorLearningLogView traverses multiple generations."""
    population = create_weighted_population(score=1.0, change_summary=None)

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0, change_summary="grandchild change")
    population.add(grandchild, grandchild_result)

    view = AncestorLearningLogView(population)
    entries = view.get_entries_for_organism(grandchild)

    # Grandchild should see its own entry and child's entry
    # Initial has no change_summary, so no entry for it
    assert len(entries) == 2
    change_summaries = [entry.attempted_change for entry in entries]
    assert "grandchild change" in change_summaries
    assert "child change" in change_summaries


def test_ancestor_learning_log_view_multiple_ancestors() -> None:
    """Test that all ancestor entries are collected."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0, change_summary="grandchild change")
    population.add(grandchild, grandchild_result)

    view = AncestorLearningLogView(population)
    entries = view.get_entries_for_organism(grandchild)

    # Should see entries from grandchild itself, child, and initial_organism
    assert len(entries) == 3
    change_summaries = [entry.attempted_change for entry in entries]
    assert "grandchild change" in change_summaries
    assert "child change" in change_summaries
    assert "root change" in change_summaries


def test_ancestor_learning_log_view_max_depth() -> None:
    """Test that max_depth limits ancestor traversal."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0, change_summary="grandchild change")
    population.add(grandchild, grandchild_result)

    great_grandchild, great_grandchild_result = create_test_organism(
        parent=grandchild, score=2.5, change_summary="great-grandchild change"
    )
    population.add(great_grandchild, great_grandchild_result)

    # Create view with max_depth=1
    # Depth 0 = great_grandchild itself, depth 1 = grandchild (parent)
    view = AncestorLearningLogView(population, max_depth=1)
    entries = view.get_entries_for_organism(great_grandchild)

    # Should see great-grandchild (depth 0) and grandchild (depth 1)
    assert len(entries) == 2
    change_summaries = [entry.attempted_change for entry in entries]
    assert "great-grandchild change" in change_summaries
    assert "grandchild change" in change_summaries
    # Should NOT see child or root change
    assert "child change" not in change_summaries
    assert "root change" not in change_summaries


def test_ancestor_learning_log_view_max_depth_zero() -> None:
    """Test that max_depth=0 only includes the organism itself."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    view = AncestorLearningLogView(population, max_depth=0)
    entries = view.get_entries_for_organism(child)

    # Should only see child's own entry
    assert len(entries) == 1
    assert entries[0].attempted_change == "child change"


def test_ancestor_learning_log_view_no_entries() -> None:
    """Test when ancestors have no learning log entries."""
    population = create_weighted_population(score=1.0, change_summary=None)

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary=None)

    view = AncestorLearningLogView(population)
    entries = view.get_entries_for_organism(child)

    # No change summaries, so no learning log entries
    assert entries == []


# ============================================================================
# NeighborhoodLearningLogView Tests
# ============================================================================


def test_neighborhood_learning_log_view_distance_0() -> None:
    """Test NeighborhoodLearningLogView with max_distance=0 (only self)."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    view = NeighborhoodLearningLogView(population, max_distance=0)
    entries = view.get_entries_for_organism(initial_organism)

    # Distance 0 should only include the organism itself
    assert len(entries) == 1


def test_neighborhood_learning_log_view_distance_1() -> None:
    """Test NeighborhoodLearningLogView with immediate neighbors."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child1, child1_result = create_test_organism(parent=initial_organism, score=1.5, change_summary="child1 change")
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=initial_organism, score=1.6, change_summary="child2 change")
    population.add(child2, child2_result)

    view = NeighborhoodLearningLogView(population, max_distance=1)
    entries = view.get_entries_for_organism(child1)

    # child1 at distance 0, initial_organism (parent) at distance 1, child2 at distance 2 (through parent)
    # So should see child1 and initial_organism
    change_summaries = {entry.attempted_change for entry in entries}
    assert change_summaries == {"child1 change", "root change"}


def test_neighborhood_learning_log_view_with_children() -> None:
    """Test that neighborhood includes children."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child1, child1_result = create_test_organism(parent=initial_organism, score=1.5, change_summary="child1 change")
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=initial_organism, score=1.6, change_summary="child2 change")
    population.add(child2, child2_result)

    child3, child3_result = create_test_organism(parent=child2, score=1.6, change_summary="child3 change")
    population.add(child3, child3_result)

    view = NeighborhoodLearningLogView(population, max_distance=1)
    entries = view.get_entries_for_organism(initial_organism)

    # From initial_organism: child1 and child2 are at distance 1
    change_summaries = {entry.attempted_change for entry in entries}
    assert change_summaries == {"root change", "child1 change", "child2 change"}


def test_neighborhood_learning_log_view_with_siblings() -> None:
    """Test that neighborhood includes siblings (through parent)."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child1, child1_result = create_test_organism(parent=initial_organism, score=1.5, change_summary="child1 change")
    population.add(child1, child1_result)

    child2, child2_result = create_test_organism(parent=initial_organism, score=1.6, change_summary="child2 change")
    population.add(child2, child2_result)

    child3, child3_result = create_test_organism(parent=child2, score=1.6, change_summary="child3 change")
    population.add(child3, child3_result)

    view = NeighborhoodLearningLogView(population, max_distance=2)
    entries = view.get_entries_for_organism(child1)

    # From child1: parent at distance 1, child2 (sibling) at distance 2
    change_summaries = {entry.attempted_change for entry in entries}
    # Should see sibling's change at distance 2
    assert change_summaries == {"root change", "child1 change", "child2 change"}


def test_neighborhood_learning_log_view_large_distance() -> None:
    """Test with large max_distance traverses entire tree."""
    population = create_weighted_population(score=1.0, change_summary="root change")

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary="child change")

    grandchild, grandchild_result = create_test_organism(parent=child, score=2.0, change_summary="grandchild change")
    population.add(grandchild, grandchild_result)

    view = NeighborhoodLearningLogView(population, max_distance=100)
    entries = view.get_entries_for_organism(child)

    # Should see all entries in the tree
    change_summaries = {entry.attempted_change for entry in entries}
    assert change_summaries == {"root change", "child change", "grandchild change"}


def test_neighborhood_learning_log_view_empty_entries() -> None:
    """Test when organisms have no learning log entries."""
    population = create_weighted_population(score=1.0, change_summary=None)

    initial_organism = population.organisms[0][0]

    child, child_result = add_test_child(population, initial_organism, score=1.5, change_summary=None)

    view = NeighborhoodLearningLogView(population, max_distance=2)
    entries = view.get_entries_for_organism(child)

    # No change summaries means no learning log entries
    assert entries == []


def test_neighborhood_learning_log_view_deep_tree() -> None:
    """Test neighborhood with deep tree structure."""
    population = create_weighted_population(score=1.0, change_summary="gen0")

    initial_organism = population.organisms[0][0]

    current = initial_organism
    for i in range(1, 6):
        child, child_result = create_test_organism(parent=current, score=1.0 + i * 0.1, change_summary=f"gen{i}")
        population.add(child, child_result)
        current = child

    # Test from middle organism with distance 2
    middle_organisms = [org for org, _ in population.organisms if org.from_change_summary == "gen3"]
    assert len(middle_organisms) == 1
    middle = middle_organisms[0]

    view = NeighborhoodLearningLogView(population, max_distance=2)
    entries = view.get_entries_for_organism(middle)

    change_summaries = {entry.attempted_change for entry in entries}
    assert change_summaries == {"gen1", "gen2", "gen3", "gen4", "gen5"}


def test_neighborhood_learning_log_view_branching_tree() -> None:
    """Test neighborhood with branching tree structure."""
    population = create_weighted_population(score=1.0, change_summary="root")

    initial_organism = population.organisms[0][0]

    # Create two branches
    left, left_result = create_test_organism(parent=initial_organism, score=1.5, change_summary="left")
    population.add(left, left_result)

    right, right_result = create_test_organism(parent=initial_organism, score=1.6, change_summary="right")
    population.add(right, right_result)

    left_child, left_child_result = create_test_organism(parent=left, score=2.0, change_summary="left_child")
    population.add(left_child, left_child_result)

    right_child, right_child_result = create_test_organism(parent=right, score=2.1, change_summary="right_child")
    population.add(right_child, right_child_result)

    view = NeighborhoodLearningLogView(population, max_distance=3)
    entries = view.get_entries_for_organism(left_child)

    change_summaries = {entry.attempted_change for entry in entries}
    # From left_child at distance 0: can reach left (1), root (2), right (3), right_child (4)
    # So with distance=3, should see left_child, left, root, and right
    assert change_summaries == {"left_child", "left", "root", "right"}


def test_learning_log_view_initialization() -> None:
    """Test that all learning log views can be initialized with population."""
    population = create_weighted_population(score=1.0)

    # All should initialize without error
    empty_view = EmptyLearningLogView(population)
    assert empty_view._learning_log == population.learning_log
    assert empty_view._population == population

    ancestor_view = AncestorLearningLogView(population)
    assert ancestor_view._learning_log == population.learning_log
    assert ancestor_view._population == population

    neighborhood_view = NeighborhoodLearningLogView(population, max_distance=1)
    assert neighborhood_view._learning_log == population.learning_log
    assert neighborhood_view._population == population
