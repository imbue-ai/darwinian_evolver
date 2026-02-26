import argparse
import concurrent
import json
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from darwinian_evolver.cli_common import parse_learning_log_view_type
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.population import Population
from darwinian_evolver.problems.arc_agi import ArcAgiEvaluationFailureCase
from darwinian_evolver.problems.arc_agi import USE_PROVIDER
from darwinian_evolver.problems.arc_agi import get_current_cost
from darwinian_evolver.problems.arc_agi import make_arc_agi_problem
from darwinian_evolver.problems.arc_agi_poetiq import build_kaggle_two_attempts
from darwinian_evolver.problems.arc_agi_poetiq import coerce_grid
from darwinian_evolver.problems.arc_agi_poetiq import grids_equal
from darwinian_evolver.problems.arc_agi_poetiq import score_task

if USE_PROVIDER == "openai":
    # GPT 5.2 tends to assign lower transfer scores than Gemini 3.
    SUFFICIENT_TRANSFER_SCORE = 0.7
elif USE_PROVIDER == "anthropic":
    # Opus 4.6 is even more conservative with transfer scores...
    SUFFICIENT_TRANSFER_SCORE = 0.35
else:
    # This value is calibrated primarily for Gemini 3 models
    SUFFICIENT_TRANSFER_SCORE = 0.95

MIDPOINT_SCORE_PERCENTILE = 99.0

USE_EXTENSIVE_SEARCH = True


def _has_hit_stopping_criteria(population: Population) -> bool:
    complete_example_solutions = [
        eval_result for _, eval_result in population.organisms if eval_result.correctness_score > 0.999
    ]
    complete_example_solutions.sort(key=lambda er: er.score, reverse=True)
    complete_solutions = [
        eval_result
        for _, eval_result in population.organisms
        if (eval_result.correctness_score > 0.999 and eval_result.transfer_score >= SUFFICIENT_TRANSFER_SCORE)
    ]
    complete_solutions.sort(key=lambda er: er.score, reverse=True)
    if USE_EXTENSIVE_SEARCH:
        # At least one solution with a sufficiently high transfer score, and either:
        if len(complete_solutions) < 1:
            return False

        # a) all solutions that agree on the examples also agree on the challenges,
        all_complete_example_solutions_agree = len(complete_example_solutions) >= 2 and all(
            all(
                grids_equal(coerce_grid(fc1.output), coerce_grid(fc2.output))
                for fc1, fc2 in zip(
                    other_complete_example_solution.holdout_failure_cases,
                    complete_example_solutions[0].holdout_failure_cases,
                )
            )
            for other_complete_example_solution in complete_example_solutions[1:]
        )

        # b) or we must have at least 2 different predictions with high transfer scores for each challenge, AND two that agree.
        has_two_different_solutions = all(
            any(
                not grids_equal(
                    coerce_grid(other_complete_solution.holdout_failure_cases[idx].output), coerce_grid(fc2.output)
                )
                for other_complete_solution in complete_solutions[1:]
            )
            for idx, fc2 in enumerate(complete_solutions[0].holdout_failure_cases)
        )

        two_complete_solutions_agree = False
        for idx, reference_solution in enumerate(complete_solutions):
            another_complete_solution_agrees = any(
                all(
                    grids_equal(
                        coerce_grid(fc1.output),
                        coerce_grid(fc2.output),
                    )
                    for fc1, fc2 in zip(
                        reference_solution.holdout_failure_cases, other_complete_solution.holdout_failure_cases
                    )
                )
                for other_complete_solution in complete_solutions[:idx] + complete_solutions[idx + 1 :]
            )
            two_complete_solutions_agree = two_complete_solutions_agree or another_complete_solution_agrees

        return all_complete_example_solutions_agree or (has_two_different_solutions and two_complete_solutions_agree)
    else:
        # Stop if we have at least two solutions that solve the task fully, according to their correctness and transfer scores.
        return len(complete_solutions) >= 2


def _eval_task_data(
    task_id: str,
    task: dict,
    log_path: Path,
    max_iterations: int,
    extra_iterations_after_solution: int,
    num_parents_per_iteration: int,
    learning_log_type: str = "none",
    gt_outputs: list | None = None,
    verbose: bool = False,
    max_time_seconds: int | None = None,
    include_crossover: bool = False,
    crossover_frequency: float = 0.25,
    crossover_min_population: int = 3,
) -> tuple[str, list[dict]]:
    if log_path.exists():
        log_path.unlink()

    problem = make_arc_agi_problem(
        task,
        gt_outputs=gt_outputs,
        include_crossover=include_crossover,
        crossover_frequency=crossover_frequency,
        crossover_min_population_size=crossover_min_population,
    )

    evolve_loop = EvolveProblemLoop(
        problem,
        learning_log_view_type=parse_learning_log_view_type(learning_log_type),
        sharpness=10.0,
        midpoint_score_percentile=MIDPOINT_SCORE_PERCENTILE,
        novelty_weight=0.2,
        batch_size=32,
        should_verify_mutations=True,
        num_parents_per_iteration=num_parents_per_iteration,
    )

    start_time = time.time()
    best_organism_result = None
    previous_best_result = None
    remaining_iterations = max_iterations
    while remaining_iterations:
        for snapshot in evolve_loop.run(1):
            best_organism_result = snapshot.best_organism_result
            with log_path.open("a", encoding="utf-8") as f:
                log_dict = {
                    "iteration": snapshot.iteration,
                    "population": snapshot.population_json_log,
                    "verification_failures": snapshot.population_json_log.get("organisms_failed_verification", []),
                }
                f.write(json.dumps(log_dict) + "\n")
            if verbose:
                if previous_best_result is not None:
                    improvement = best_organism_result[1].score - previous_best_result[1].score
                else:
                    improvement = 0
                print(
                    f"[{task_id}] Iteration {snapshot.iteration} score {best_organism_result[1].score:.4f}"
                    + (f" (+{improvement:.4f})" if improvement else " (~)"),
                )
            previous_best_result = best_organism_result

        remaining_iterations -= 1

        # Keep iterating until we either hit max_iterations, or hit the early stopping condition.
        # In the latter case, we iterate up to extra_iterations_after_solution more times to see if we can find an even better solution.
        has_reached_stopping_criteria = _has_hit_stopping_criteria(evolve_loop.population)
        if has_reached_stopping_criteria:
            remaining_iterations = min(remaining_iterations, extra_iterations_after_solution)

        if max_time_seconds is not None and (time.time() - start_time) > max_time_seconds:
            if verbose:
                print(f"[{task_id}] Reached max time of {max_time_seconds} seconds; stopping early.")
            remaining_iterations = 0

    assert best_organism_result is not None
    best_holdout_results = [best_organism_result[1].holdout_failure_cases]

    alternative_holdout_results = _select_alternative_attempts(
        best_organism_result[1].holdout_failure_cases, evolve_loop.population
    )
    best_holdout_results.append(alternative_holdout_results)
    test_in = [ex["input"] for ex in task["test"]]

    preds = build_kaggle_two_attempts(
        results=best_holdout_results,
        test_in=test_in,
    )

    return task_id, preds


def _select_alternative_attempts(
    best_results: list[ArcAgiEvaluationFailureCase], population: Population
) -> list[ArcAgiEvaluationFailureCase]:
    # For the second attempt, find the highest-scoring organism that produces different outputs on the test input(s).
    alternative_results = best_results.copy()
    sorted_population = sorted(population.organisms, key=lambda oe: oe[1].score, reverse=True)
    for idx, fc in enumerate(best_results):
        fc_grid = coerce_grid(fc.output)
        for _, eval_result in sorted_population:
            organism_test_output = [
                afc for afc in eval_result.holdout_failure_cases if afc.data_point_id == fc.data_point_id
            ][0]
            afc_grid = coerce_grid(organism_test_output.output)
            if afc_grid and not grids_equal(fc_grid, afc_grid):
                alternative_results[idx] = organism_test_output
                break

    return alternative_results


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    arg_parser = argparse.ArgumentParser(
        description="Use the Darwinian Evolver to solve ARC-AGI tasks and produce Kaggle submission file."
    )
    arg_parser.add_argument(
        "--challenges",
        type=str,
        help="Path to ARC-AGI evaluation challenges JSON file.",
        required=True,
    )
    arg_parser.add_argument(
        "--solutions",
        type=str,
        help="Path to ARC-AGI evaluation solutions JSON file.",
        required=False,
    )
    arg_parser.add_argument(
        "--hide_solutions",
        action="store_true",
        help="As an additional safeguard, hide the solutions from the solver to make sure that we're not accidentally leaking them into the solution process.",
        required=False,
    )
    arg_parser.add_argument(
        "--num_problems",
        type=int,
        help="Number of problems to evaluate (from start of file, or randomly if --shuffle_problems is set). If not set, evaluates all problems.",
        required=False,
    )
    arg_parser.add_argument(
        "--shuffle_problems",
        action="store_true",
        help="Randomly select problems instead of taking the first N. Can be combined with --num_problems to select N random problems.",
        required=False,
    )
    arg_parser.add_argument(
        "--problem_ids",
        type=str,
        nargs="+",
        help="Specific problem IDs to evaluate. If not set, evaluates all problems.",
        required=False,
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to output submission and logs to.",
        required=True,
    )
    arg_parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Whether to overwrite existing output files.",
        required=False,
    )
    arg_parser.add_argument(
        "--max_iterations",
        type=int,
        help="Maximum number of evolution iterations per problem.",
        default=16,
        required=False,
    )
    arg_parser.add_argument(
        "--extra_iterations_after_solution",
        type=int,
        help="Extra iterations to run after finding a full solution to see if we can find a more general one.",
        default=0,
        required=False,
    )
    arg_parser.add_argument(
        "--num_parents_per_iteration",
        type=int,
        help="Number of parent organisms to select per iteration.",
        # 2 is more efficient for strong models (Opus 4.6 etc). Can be increased to 3-4 for cheaper/weaker models (such as Gemini 3 Flash)
        default=2,
        required=False,
    )
    arg_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of tasks to evaluate in parallel.",
        default=32,
        required=False,
    )
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose output during evaluation.",
        required=False,
    )
    arg_parser.add_argument(
        "--max_time",
        type=int,
        help="Maximum time in seconds to spend per problem.",
        default=60 * 60 * 5,  # 5 hours
        required=False,
    )
    arg_parser.add_argument(
        "--learning_log",
        type=str,
        help="Type of learning log view to provide to the evolver (none, ancestors, neighborhood-3, etc.).",
        default="none",
    )
    arg_parser.add_argument(
        "--include_crossover",
        action="store_true",
        default=True,
        help="Whether to include crossover mutator that combines insights from multiple parent organisms.",
        required=False,
    )
    arg_parser.add_argument(
        "--crossover_frequency",
        type=float,
        default=0.25,
        help="Probability of performing crossover mutation (when crossover is enabled). Default: 0.25",
        required=False,
    )
    arg_parser.add_argument(
        "--crossover_min_population",
        type=int,
        default=3,
        help="Minimum population size required before crossover mutations are attempted. Default: 3",
        required=False,
    )
    args = arg_parser.parse_args()

    # Load challenges
    with open(args.challenges, "r", encoding="utf-8") as f:
        challenges_blob: dict[str, dict] = json.load(f)

    # Load solutions if present; disable scoring if missing/unreadable
    solutions_blob: dict[str, list] | None = None
    if args.solutions:
        try:
            with open(args.solutions, "r", encoding="utf-8") as f:
                solutions_blob = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load solutions file '{args.solutions}': {e}\nScoring will be disabled.")

    output_dir = Path(args.output_dir)
    output_logs_dir = output_dir / "logs"
    output_submission = output_dir / "kaggle_submission.json"
    output_dir.mkdir(parents=True, exist_ok=args.overwrite_output)
    output_logs_dir.mkdir(parents=True, exist_ok=True)

    items = list(challenges_blob.items())

    if args.shuffle_problems:
        random.shuffle(items)

    if args.num_problems is not None:
        items = items[: args.num_problems]

    if args.problem_ids is not None:
        items = [item for item in items if item[0] in args.problem_ids]

    submission: dict[str, list[dict]] = {}

    # running scores only if solutions available
    per_task_scores: dict[str, float] = {}
    total = 0
    correct = 0.0
    incorrect = 0.0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        tasks = []
        for task_id, task in items:
            log_path = output_logs_dir / f"{task_id}.jsonl"
            if not args.hide_solutions and solutions_blob is not None and task_id in solutions_blob:
                gt_outputs = solutions_blob[task_id]
            else:
                gt_outputs = None
            tasks.append(
                executor.submit(
                    _eval_task_data,
                    task_id,
                    task,
                    log_path=log_path,
                    max_iterations=args.max_iterations,
                    extra_iterations_after_solution=args.extra_iterations_after_solution,
                    learning_log_type=args.learning_log,
                    gt_outputs=gt_outputs,
                    verbose=args.verbose,
                    max_time_seconds=args.max_time,
                    include_crossover=args.include_crossover,
                    crossover_frequency=args.crossover_frequency,
                    crossover_min_population=args.crossover_min_population,
                    num_parents_per_iteration=args.num_parents_per_iteration,
                )
            )

        for future in concurrent.futures.as_completed(tasks):
            try:
                task_id, preds = future.result()
            except Exception as e:
                print(f"Task evaluation failed: {e}")
                continue

            submission[task_id] = preds

            # running scores if solutions available
            if solutions_blob is not None and task_id in solutions_blob:
                gt_outputs = solutions_blob[task_id]
                task_score = score_task(preds, gt_outputs)
                per_task_scores[task_id] = task_score
                total += 1
                correct += task_score
                incorrect += 1 - task_score
                mark = "✓" if task_score == 1.0 else "✗"
                print(f"{mark} {task_id} [{correct}/{total}, ${get_current_cost():.2f}]")
            else:
                print(f"· {task_id} [${get_current_cost():.2f}]")

            # write cumulative Kaggle output after each task
            try:
                with output_submission.open("w", encoding="utf-8") as f:
                    json.dump(submission, f)
            except Exception as e:
                print(f"WARNING: Failed to write partial output to {output_submission}: {e}")

    print("\n=== Summary ===")
    print(f"Data file: {args.challenges}")
    print(f"Problems: {len(items)}")
    print(f"Total cost: ${get_current_cost()}")
    if solutions_blob is not None and per_task_scores:
        acc = correct / total
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {acc * 100:.3f}")
    else:
        print("Scoring: disabled or no tasks matched in solutions.")
