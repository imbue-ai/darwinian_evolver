"""A script for performing hyperparameter sweeps with the evolver."""

import argparse
import concurrent
import enum
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from tqdm import tqdm

from darwinian_evolver.cli_common import HyperparameterConfig
from darwinian_evolver.cli_common import build_hyperparameter_config_from_args
from darwinian_evolver.cli_common import parse_learning_log_view_type
from darwinian_evolver.cli_common import parse_midpoint_score
from darwinian_evolver.cli_common import register_hyperparameter_args
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.evolver import EvolverStats
from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.registry import AVAILABLE_PROBLEMS


class Hyperparameter(enum.Enum):
    BATCH_SIZE = "batch_size"
    VERIFY_MUTATIONS = "verify_mutations"
    NUM_PARENTS_PER_ITERATION = "num_parents_per_iteration"
    SHARPNESS_AND_MIDPOINT_SCORE = "sharpness_and_midpoint_score"
    NOVELTY_WEIGHT = "novelty_weight"
    LEARNING_LOG_VIEW_TYPE = "learning_log_view_type"


class IterationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    iteration: int
    evolver_stats: EvolverStats
    score_percentiles: dict[float, float]
    population_size: int


class AttemptResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: HyperparameterConfig
    attempt: int
    iterations: list[IterationResult]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the hyperparameter sweep."""
    arg_parser = argparse.ArgumentParser(
        description="Evaluate hyperparameter configurations for the Darwinian Evolver.",
        epilog="Example (evaluate different batch_size values): eval_hyperparameters.py --csv_output batch_size.csv multiplication_verifier batch_size 1 2 4 8",
    )
    arg_parser.add_argument(
        "problem",
        type=str,
        choices=AVAILABLE_PROBLEMS.keys(),
        help="The problem to evolve. Available problems: " + ", ".join(AVAILABLE_PROBLEMS.keys()),
    )
    arg_parser.add_argument(
        "parameter",
        type=str,
        choices=[member.value for member in Hyperparameter],
        help="The parameter to sweep.",
    )
    arg_parser.add_argument(
        "values",
        type=str,
        nargs="+",
        help="The values to evaluate for the selected parameter.",
    )

    runtime_args = arg_parser.add_argument_group("Runtime")
    runtime_args.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        required=False,
        help="The number of iterations to run the evolution loop for (per attempt/configuration). Default is 10.",
    )
    runtime_args.add_argument(
        "--num_attempts",
        type=int,
        default=1,
        required=False,
        help="The number of times to run each configuration. Default is 1.",
    )
    runtime_args.add_argument(
        "--outer_concurrency",
        type=int,
        default=4,
        required=False,
        help="The number of attempts/configurations to evaluate in parallel. Default is 4.",
    )
    runtime_args.add_argument(
        "--inner_concurrency",
        type=int,
        default=4,
        required=False,
        help="The number of evaluator and mutator threads to use for one given attempt/configuration. Default is 4.",
    )

    output_args = arg_parser.add_argument_group("Output")
    output_args.add_argument(
        "--csv_output",
        type=str,
        required=True,
        help="The path to the CSV file to output results.",
    )

    hyperparameter_args = arg_parser.add_argument_group("Baseline parameters")
    register_hyperparameter_args(hyperparameter_args)

    return arg_parser.parse_args()


def build_config_update(parameter: Hyperparameter, value: str) -> dict:
    """Build the configuration update for a given hyperparameter and value."""
    match parameter:
        case Hyperparameter.BATCH_SIZE:
            return {"batch_size": int(value)}
        case Hyperparameter.VERIFY_MUTATIONS:
            return {"verify_mutations": value.lower() == "true"}
        case Hyperparameter.NUM_PARENTS_PER_ITERATION:
            return {"num_parents_per_iteration": int(value)}
        case Hyperparameter.SHARPNESS_AND_MIDPOINT_SCORE:
            sharpness_str, midpoint_score_str = value.split(",")
            fixed_midpoint_score, midpoint_score_percentile = parse_midpoint_score(midpoint_score_str)
            return {
                "sharpness": float(sharpness_str),
                "fixed_midpoint_score": fixed_midpoint_score,
                "midpoint_score_percentile": midpoint_score_percentile,
            }
        case Hyperparameter.NOVELTY_WEIGHT:
            return {"novelty_weight": float(value)}
        case Hyperparameter.LEARNING_LOG_VIEW_TYPE:
            return {"learning_log_view_type": value}
        case _:
            raise ValueError(f"Unknown parameter: {parameter}")


def run_evolve_loop(
    problem: Problem,
    config: HyperparameterConfig,
    num_iterations: int,
    progress_bar: tqdm,
    label: str,
    attempt: int,
    inner_concurrency: int,
) -> AttemptResult:
    """Run the evolution loop for a given problem and configuration."""
    evolve_loop = EvolveProblemLoop(
        problem,
        learning_log_view_type=parse_learning_log_view_type(config.learning_log_view_type),
        num_parents_per_iteration=config.num_parents_per_iteration,
        fixed_midpoint_score=config.fixed_midpoint_score,
        midpoint_score_percentile=config.midpoint_score_percentile,
        sharpness=config.sharpness,
        batch_size=config.batch_size,
        should_verify_mutations=config.verify_mutations,
        mutator_concurrency=inner_concurrency,
        evaluator_concurrency=inner_concurrency,
    )

    iteration_results = []
    evolver_stats_accumulator = EvolverStats()
    remaining_steps = num_iterations + 1
    try:
        with tqdm(total=remaining_steps, desc=f"  {label}", leave=False, colour="blue") as attempt_progress_bar:
            for snapshot in evolve_loop.run(num_iterations=num_iterations):
                evolver_stats_accumulator += snapshot.evolver_stats

                iteration_results.append(
                    IterationResult(
                        iteration=snapshot.iteration,
                        evolver_stats=evolver_stats_accumulator.model_copy(),
                        score_percentiles=snapshot.score_percentiles,
                        population_size=snapshot.population_size,
                    )
                )
                progress_bar.update()
                attempt_progress_bar.update()
                remaining_steps -= 1
    finally:
        progress_bar.update(remaining_steps)

    return AttemptResult(config=config, iterations=iteration_results, attempt=attempt)


if __name__ == "__main__":
    args = parse_args()

    # Select the specified problem
    problem = AVAILABLE_PROBLEMS[args.problem]()

    # Create base configuration from command line arguments
    base_config = build_hyperparameter_config_from_args(args)

    # Run the evolution loop for each configuration and attempt
    total_num_iterations = args.num_attempts * (args.num_iterations + 1) * len(args.values)
    with tqdm(total=total_num_iterations, desc="Evaluating", colour="green") as progress_bar:
        with ThreadPoolExecutor(max_workers=args.outer_concurrency) as executor:
            attempt_futures = []
            for value in args.values:
                config = base_config.model_copy(update=build_config_update(Hyperparameter(args.parameter), value))
                for attempt in range(args.num_attempts):
                    label = f"{args.parameter}={value} ({attempt + 1}/{args.num_attempts})"
                    future = executor.submit(
                        run_evolve_loop,
                        problem,
                        config,
                        args.num_iterations,
                        progress_bar=progress_bar,
                        label=label,
                        attempt=attempt,
                        inner_concurrency=args.inner_concurrency,
                    )
                    attempt_futures.append(future)

            gathered_results = []
            for future in concurrent.futures.as_completed(attempt_futures):
                try:
                    gathered_results.append(future.result())
                except Exception as e:
                    tqdm.write(f"Attempt failed: {e}")

    # Report the results
    if args.csv_output:
        rows = []
        for attempt_result in gathered_results:
            config_dict = attempt_result.config.model_dump()
            for iteration_result in attempt_result.iterations:
                # Flatten out all fields into the row
                row = {}
                row.update(config_dict)
                row["attempt"] = attempt_result.attempt
                row["iteration"] = iteration_result.iteration
                for percentile, score in iteration_result.score_percentiles.items():
                    row[f"score_p{percentile:g}"] = score
                row["population_size"] = iteration_result.population_size
                row.update(iteration_result.evolver_stats.model_dump())
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(args.csv_output, index=False)
        print(f"Results saved to {args.csv_output}")

    # TODO (danielmewes): Implement W&B reporting support
