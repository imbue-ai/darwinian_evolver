"""Shared utility classes and functions for Darwinian Evolver command line scripts."""

import argparse

from pydantic import BaseModel

from darwinian_evolver.learning_log_view import AncestorLearningLogView
from darwinian_evolver.learning_log_view import EmptyLearningLogView
from darwinian_evolver.learning_log_view import LearningLogView
from darwinian_evolver.learning_log_view import NeighborhoodLearningLogView


class HyperparameterConfig(BaseModel):
    batch_size: int
    verify_mutations: bool
    num_parents_per_iteration: int
    sharpness: float | None
    fixed_midpoint_score: float | None
    midpoint_score_percentile: float | None
    novelty_weight: float | None
    learning_log_view_type: str


def register_hyperparameter_args(arg_container: argparse._ActionsContainer) -> None:
    """Register hyperparameter arguments for the given argument parser."""
    arg_container.add_argument(
        "--num_parents_per_iteration",
        type=int,
        default=4,
        required=False,
        help="The number of parents to select for each iteration. Default is 4.",
    )
    arg_container.add_argument(
        "--midpoint_score",
        type=str,
        default=None,
        required=False,
        help="The midpoint score to use for parent sampling. Can be a float value or 'pXX' where XX is a percentile (0-100) to track dynamically. Default is 'p75' (75th percentile).",
    )
    arg_container.add_argument(
        "--sharpness",
        type=float,
        default=None,
        required=False,
        help="The sharpness parameter for the sigmoid function used in parent selection. Default is 10.0.",
    )
    arg_container.add_argument(
        "--novelty_weight",
        type=float,
        default=None,
        required=False,
        help="The weight of novelty in the selection process. Default is 1.0.",
    )
    arg_container.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="The number of failure cases to pass into each mutator application at once. Default is 1.",
    )
    arg_container.add_argument(
        "--verify_mutations",
        action="store_true",
        default=False,
        required=False,
        help="Verify mutations before adding them to the population. Only mutations that improve on the given failure cases will be accepted.",
    )
    arg_container.add_argument(
        "--learning_log",
        type=str,
        default="none",
        required=False,
        help="The type of learning log to use. Options are: 'none', 'ancestors', 'neighborhood-N' (where N is an integer distance). Default is 'none'.",
    )


def parse_midpoint_score(midpoint_score_str: str) -> tuple[float | None, float | None]:
    """
    Parse midpoint score argument.

    Returns (fixed_score, percentile) where exactly one is None
    """
    if midpoint_score_str.lower().startswith("p"):
        percentile = float(midpoint_score_str[1:])
        if not (0 <= percentile <= 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        return None, percentile
    else:
        fixed_score = float(midpoint_score_str)
        return fixed_score, None


def build_hyperparameter_config_from_args(args: argparse.Namespace) -> HyperparameterConfig:
    """Build a HyperparameterConfig from the command line arguments."""
    if args.midpoint_score:
        fixed_midpoint_score, midpoint_score_percentile = parse_midpoint_score(args.midpoint_score)
    else:
        fixed_midpoint_score = None
        midpoint_score_percentile = None

    return HyperparameterConfig(
        batch_size=args.batch_size,
        verify_mutations=args.verify_mutations,
        num_parents_per_iteration=args.num_parents_per_iteration,
        sharpness=args.sharpness,
        fixed_midpoint_score=fixed_midpoint_score,
        midpoint_score_percentile=midpoint_score_percentile,
        novelty_weight=args.novelty_weight,
        learning_log_view_type=args.learning_log,
    )


def parse_learning_log_view_type(view_type_str: str) -> tuple[type[LearningLogView], dict[str, any]]:
    """
    Parse learning log view type from string.

    Returns the class of the LearningLogView and any additional keyword parameters as a dictionary that need
    to be passed to the constructor.
    """
    if view_type_str.lower() == "none":
        return EmptyLearningLogView, {}
    elif view_type_str.lower() == "ancestors":
        return AncestorLearningLogView, {}
    elif view_type_str.lower().startswith("neighborhood-"):
        distance_str = view_type_str[len("neighborhood-") :]
        try:
            max_distance = int(distance_str)
        except ValueError:
            raise ValueError(f"Invalid neighborhood distance: {distance_str}")
        return NeighborhoodLearningLogView, {"max_distance": max_distance}
    else:
        raise ValueError(
            f"Invalid learning log view type: {view_type_str}. Valid types are: none, ancestors, neighborhood-N (where N is an integer)."
        )
