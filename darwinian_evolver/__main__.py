import argparse
import json
import sys
import tempfile
from pathlib import Path

from darwinian_evolver.cli_common import build_hyperparameter_config_from_args
from darwinian_evolver.cli_common import parse_learning_log_view_type
from darwinian_evolver.cli_common import register_hyperparameter_args
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.evolve_problem_loop import IterationSnapshot
from darwinian_evolver.problems.registry import AVAILABLE_PROBLEMS
from darwinian_evolver.storage import upload_bytes_to_s3
from darwinian_evolver.storage import upload_file_to_s3_fixed_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Darwin Prompt evolution loop."""
    arg_parser = argparse.ArgumentParser(description="Run the Darwin Prompt evolution loop on a given problem.")
    arg_parser.add_argument(
        "problem",
        type=str,
        choices=AVAILABLE_PROBLEMS.keys(),
        help="The problem to evolve. Available problems: " + ", ".join(AVAILABLE_PROBLEMS.keys()),
    )

    hyperparameter_args = arg_parser.add_argument_group("Hyperparameters")
    register_hyperparameter_args(hyperparameter_args)

    runtime_args = arg_parser.add_argument_group("Runtime")
    runtime_args.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        required=False,
        help="The number of iterations to run the evolution loop for. Default is 5.",
    )
    runtime_args.add_argument(
        "--mutator_concurrency",
        type=int,
        default=10,
        required=False,
        help="The maximum number of mutators to run concurrently. Default is 10.",
    )
    runtime_args.add_argument(
        "--evaluator_concurrency",
        type=int,
        default=10,
        required=False,
        help="The maximum number of evaluators to run concurrently. Default is 10.",
    )
    runtime_args.add_argument(
        "--use_process_pool",
        action="store_true",
        default=False,
        help="Use multi-processing instead of multi-threading to run mutators and evaluators.",
    )

    fixed_tree_mode_args = arg_parser.add_argument_group("Fixed Tree Mode")
    fixed_tree_mode_args.add_argument(
        "--fixed_children_per_generation",
        type=int,
        nargs="+",
        required=False,
        help="Space-separated list of number of children per generation for fixed tree mode (e.g., '10 1' or '6 4 3 2 2'). When provided, enables fixed tree structure instead of weighted parent sampling. Each iteration will produce a generation. Iterations beyond the list length will cycle through the list again.",
    )

    input_output_args = arg_parser.add_argument_group("Input/Output")
    input_output_args.add_argument(
        "--resume_from_snapshot",
        type=Path,
        required=False,
        help="Path to the snapshot file to resume from.",
    )
    input_output_args.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        help="Output directory for logs and evaluator results. Will be created if it does not exist.",
    )
    input_output_args.add_argument(
        "--overwrite_snapshots",
        action="store_true",
        help="Overwrite existing snapshot files.",
    )
    input_output_args.add_argument(
        "--s3_dir",
        type=Path,
        required=False,
        help="S3 path to upload results to, relative to the s3://int8-shared-internal/ bucket.",
    )

    return arg_parser.parse_args()


def print_snapshot_summary(snapshot: IterationSnapshot) -> None:
    print(f"Iteration {snapshot.iteration}:")
    print("  Best score:", snapshot.best_organism_result[1].score)
    print(
        "  Best organism:",
        snapshot.best_organism_result[0].id,
        str(
            snapshot.best_organism_result[0].model_dump(
                exclude={
                    # Exclude all the fields from the Organism base class to highlight the problem-specific fields
                    "parent",
                    "additional_parents",
                    "id",
                    "from_failure_cases",
                    "from_learning_log_entries",
                    "from_change_summary",
                    "visualizer_props",
                }
            )
        )[:80]
        + "...",
    )
    print("  Population size:", snapshot.population_size)
    print("  Evolver stats: ")
    for s, v in snapshot.evolver_stats.model_dump().items():
        print(f"    {s}: {v}")


if __name__ == "__main__":
    args = parse_args()

    # Select the specified problem
    problem = AVAILABLE_PROBLEMS[args.problem]()

    if args.batch_size > 1 and not any(mutator.supports_batch_mutation for mutator in problem.mutators):
        print(
            "Warning: Batch size is set to greater than 1, but no mutators of this problem support batch mutation. Batch size will have no effect."
        )

    # Validate fixed_children_per_generation if provided
    fixed_children_per_generation = args.fixed_children_per_generation
    if fixed_children_per_generation is not None:
        if any(x <= 0 for x in fixed_children_per_generation):
            print("Error: All values in --fixed_children_per_generation must be positive")
            sys.exit(1)

    hyperparameter_config = build_hyperparameter_config_from_args(args)

    # Setup output directories
    snapshot_dir = None
    json_log_file = None
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_log_file = args.output_dir / "results.jsonl"
        snapshot_dir = args.output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        problem.evaluator.set_output_dir(str(args.output_dir))
    elif args.s3_dir:
        # Create temp file for results.jsonl when uploading to S3 but no local output_dir
        json_log_file = Path(tempfile.mktemp(suffix=".jsonl"))

    if args.s3_dir:
        problem.evaluator.set_s3_dir(str(args.s3_dir))

    evolve_loop = EvolveProblemLoop(
        problem,
        learning_log_view_type=parse_learning_log_view_type(hyperparameter_config.learning_log_view_type),
        num_parents_per_iteration=hyperparameter_config.num_parents_per_iteration,
        mutator_concurrency=args.mutator_concurrency,
        evaluator_concurrency=args.evaluator_concurrency,
        snapshot_to_resume_from=args.resume_from_snapshot.read_bytes() if args.resume_from_snapshot else None,
        fixed_midpoint_score=hyperparameter_config.fixed_midpoint_score,
        midpoint_score_percentile=hyperparameter_config.midpoint_score_percentile,
        sharpness=hyperparameter_config.sharpness,
        novelty_weight=hyperparameter_config.novelty_weight,
        batch_size=hyperparameter_config.batch_size,
        should_verify_mutations=hyperparameter_config.verify_mutations,
        fixed_children_per_generation=fixed_children_per_generation,
        use_process_pool_executors=args.use_process_pool,
    )

    if args.resume_from_snapshot:
        print(f"Resuming from snapshot: {args.resume_from_snapshot}")
    else:
        print("Evaluating initial organism...")

    for snapshot in evolve_loop.run(num_iterations=args.num_iterations):
        print_snapshot_summary(snapshot)

        if snapshot_dir:
            snapshot_file = snapshot_dir / f"iteration_{snapshot.iteration}.pkl"
            if snapshot_file.exists() and not args.overwrite_snapshots:
                print(f"Snapshot {snapshot_file} already exists. Use --overwrite_snapshots to overwrite.")
                sys.exit(1)

            with snapshot_file.open("wb") as file:
                file.write(snapshot.snapshot)

        if args.s3_dir:
            upload_bytes_to_s3(
                snapshot.snapshot,
                str(args.s3_dir),
                f"snapshots/iteration_{snapshot.iteration}.pkl",
            )

        if json_log_file:
            with json_log_file.open("a") as file:
                log_dict = {
                    "iteration": snapshot.iteration,
                    "population": snapshot.population_json_log,
                    "verification_failures": snapshot.population_json_log.get("organisms_failed_verification", []),
                }
                file.write(json.dumps(log_dict) + "\n")

    if args.s3_dir:
        upload_file_to_s3_fixed_path(json_log_file, str(args.s3_dir), "results.jsonl")
