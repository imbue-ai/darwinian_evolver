"""Tests for EvaluationResult methods in problem.py."""

from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.test_utils import create_test_result

# ============================================================================
# format_observed_outcome Tests
# ============================================================================


def test_format_observed_outcome_with_improvement() -> None:
    """Test format_observed_outcome when score improves over parent."""
    parent_result = create_test_result(score=1.0)

    current_result = create_test_result(score=1.5)

    outcome = current_result.format_observed_outcome(parent_result)

    assert "1.5" in outcome
    assert "improvement" in outcome.lower()
    assert "1.0" in outcome


def test_format_observed_outcome_with_regression() -> None:
    """Test format_observed_outcome when score is worse than parent."""
    parent_result = create_test_result(score=2.0)

    current_result = create_test_result(score=1.5)

    outcome = current_result.format_observed_outcome(parent_result)

    assert "1.5" in outcome
    assert "worse" in outcome.lower()
    assert "2.0" in outcome


def test_format_observed_outcome_with_no_change() -> None:
    """Test format_observed_outcome when score is same as parent."""
    parent_result = create_test_result(score=1.5)

    current_result = create_test_result(score=1.5)

    outcome = current_result.format_observed_outcome(parent_result)

    assert "1.5" in outcome
    assert "same" in outcome.lower()


def test_format_observed_outcome_with_no_rounded_change() -> None:
    """Test format_observed_outcome when score is within rounding error of parent."""
    parent_result = create_test_result(score=1.502)

    current_result = create_test_result(score=1.501)

    outcome = current_result.format_observed_outcome(parent_result)

    assert "same" in outcome.lower()
    assert "1.5" in outcome
    assert "1.502" not in outcome
    assert "1.501" not in outcome


def test_format_observed_outcome_without_parent() -> None:
    """Test format_observed_outcome when parent result is None."""
    current_result = create_test_result(score=1.5)

    outcome = current_result.format_observed_outcome(None)

    # Should only mention the current score, not comparison
    assert "1.5" in outcome
    assert "parent" not in outcome.lower()
    assert "improvement" not in outcome.lower()
    assert "worse" not in outcome.lower()


def test_format_observed_outcome_non_viable() -> None:
    """Test format_observed_outcome when organism is not viable."""
    parent_result = create_test_result(score=1.0)

    current_result = EvaluationResult(
        score=0.0,
        trainable_failure_cases=[],
        is_viable=False,
    )

    outcome = current_result.format_observed_outcome(parent_result)

    assert "Inconclusive" in outcome and "not viable" in outcome.lower()
    # Should not include score comparisons for non-viable organisms
    assert "improvement" not in outcome.lower()


def test_format_observed_outcome_non_viable_without_parent() -> None:
    """Test format_observed_outcome when organism is not viable and no parent."""
    current_result = EvaluationResult(
        score=0.0,
        trainable_failure_cases=[],
        is_viable=False,
    )

    outcome = current_result.format_observed_outcome(None)

    assert "Inconclusive" in outcome and "not viable" in outcome.lower()


# ============================================================================
# sample_trainable_failure_cases Tests
# ============================================================================


def test_sample_trainable_failure_cases_single() -> None:
    """Test sampling a single failure case."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id="fail_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_2", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_3", failure_type="type_a"),
        ],
        is_viable=True,
    )

    samples = result.sample_trainable_failure_cases(batch_size=1)

    assert len(samples) == 1
    assert samples[0] in result.trainable_failure_cases


def test_sample_trainable_failure_cases_batch() -> None:
    """Test sampling multiple failure cases."""
    result = create_test_result(score=1.0, num_trainable_failures=10, failure_type="type_a")

    samples = result.sample_trainable_failure_cases(batch_size=5)

    assert len(samples) == 5
    # All samples should be from the trainable failure cases
    for sample in samples:
        assert sample in result.trainable_failure_cases


def test_sample_trainable_failure_cases_same_type() -> None:
    """Test that all sampled failure cases have the same failure_type."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id="fail_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_2", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_3", failure_type="type_b"),
            EvaluationFailureCase(data_point_id="fail_4", failure_type="type_b"),
        ],
        is_viable=True,
    )

    samples = result.sample_trainable_failure_cases(batch_size=3)

    # All samples should have the same failure_type
    assert len(set(sample.failure_type for sample in samples)) == 1


def test_sample_trainable_failure_cases_empty() -> None:
    """Test sampling when there are no trainable failure cases."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[],
        is_viable=True,
    )

    samples = result.sample_trainable_failure_cases(batch_size=5)

    assert len(samples) == 0


def test_sample_trainable_failure_cases_batch_size_exceeds_available() -> None:
    """Test that batch_size is capped by available failure cases of the same type."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id="fail_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_2", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_3", failure_type="type_a"),
        ],
        is_viable=True,
    )

    # Request more than available
    samples = result.sample_trainable_failure_cases(batch_size=10)

    # Should return at most 3 (all available of the same type)
    assert len(samples) <= 3
    assert len(samples) > 0


def test_sample_trainable_failure_cases_multiple_types() -> None:
    """Test sampling with multiple failure types."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id="fail_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_2", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="fail_3", failure_type="type_b"),
            EvaluationFailureCase(data_point_id="fail_4", failure_type="type_b"),
        ],
        is_viable=True,
    )

    samples = result.sample_trainable_failure_cases(batch_size=2)

    # Should sample 2 cases of the same type
    assert len(samples) == 2
    assert samples[0].failure_type == samples[1].failure_type


def test_sample_trainable_failure_cases_single_failure_case() -> None:
    """Test sampling when there's only one failure case."""
    result = create_test_result(score=1.0, num_trainable_failures=1)

    samples = result.sample_trainable_failure_cases(batch_size=5)

    # Should return the single available case
    assert len(samples) == 1
    assert samples[0].data_point_id.startswith("fail_")


def test_sample_trainable_failure_cases_uniqueness() -> None:
    """Test that sampled failure cases are unique (no duplicates)."""
    result = create_test_result(score=1.0, num_trainable_failures=5, failure_type="type_a")

    samples = result.sample_trainable_failure_cases(batch_size=3)

    # Check that all samples are unique
    sample_ids = [sample.data_point_id for sample in samples]
    assert len(sample_ids) == len(set(sample_ids))


def test_sample_trainable_failure_cases_holdout_not_included() -> None:
    """Test that holdout failure cases are not included in sampling."""
    result = EvaluationResult(
        score=1.0,
        trainable_failure_cases=[
            EvaluationFailureCase(data_point_id="trainable_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="trainable_2", failure_type="type_a"),
        ],
        holdout_failure_cases=[
            EvaluationFailureCase(data_point_id="holdout_1", failure_type="type_a"),
            EvaluationFailureCase(data_point_id="holdout_2", failure_type="type_a"),
        ],
        is_viable=True,
    )

    # Sample many times
    for _ in range(20):
        samples = result.sample_trainable_failure_cases(batch_size=2)
        for sample in samples:
            # Should never get holdout cases
            assert "trainable" in sample.data_point_id
            assert "holdout" not in sample.data_point_id
