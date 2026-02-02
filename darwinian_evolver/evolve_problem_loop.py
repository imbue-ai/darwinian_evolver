import pickle
from typing import Generator

from pydantic import BaseModel
from pydantic import ConfigDict

from darwinian_evolver.evolver import Evolver
from darwinian_evolver.evolver import EvolverStats
from darwinian_evolver.learning_log_view import LearningLogView
from darwinian_evolver.population import FixedTreePopulation
from darwinian_evolver.population import Population
from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import MutatorContext
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem


class IterationSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    iteration: int
    population_size: int
    snapshot: bytes
    population_json_log: dict
    best_organism_result: tuple[Organism, EvaluationResult]
    score_percentiles: dict[float, float]
    evolver_stats: EvolverStats


class EvolveProblemLoop:
    _DEFAULT_SHARPNESS = 10.0
    _DEFAULT_MIDPOINT_SCORE_PERCENTILE = 75.0
    _DEFAULT_NOVELTY_WEIGHT = 1.0

    _problem: Problem
    _learning_log_view_type: tuple[type[LearningLogView], dict[str, any]]
    _mutator_concurrency: int
    _evaluator_concurrency: int
    _num_parents_per_iteration: int
    _fixed_midpoint_score: float | None
    _midpoint_score_percentile: float | None
    _sharpness: float | None
    _novelty_weight: float | None
    _batch_size: int
    _should_verify_mutations: bool
    _fixed_children_per_generation: list[int] | None
    _use_process_pool_executors: bool

    _current_iteration: int = 0
    _evolver: Evolver | None = None

    def __init__(
        self,
        problem: Problem,
        learning_log_view_type: tuple[type[LearningLogView], dict[str, any]],
        mutator_concurrency: int = 10,
        evaluator_concurrency: int = 10,
        num_parents_per_iteration: int = 5,
        snapshot_to_resume_from: bytes | None = None,
        fixed_midpoint_score: float | None = None,
        midpoint_score_percentile: float | None = None,
        sharpness: float | None = None,
        novelty_weight: float | None = None,
        batch_size: int = 1,
        should_verify_mutations: bool = False,
        fixed_children_per_generation: list[int] | None = None,
        use_process_pool_executors: bool = False,
    ) -> None:
        assert midpoint_score_percentile is None or fixed_midpoint_score is None, (
            "Cannot specify both fixed_midpoint_score and midpoint_score_percentile"
        )

        self._problem = problem
        self._learning_log_view_type = learning_log_view_type
        self._mutator_concurrency = mutator_concurrency
        self._evaluator_concurrency = evaluator_concurrency
        self._num_parents_per_iteration = num_parents_per_iteration
        self._fixed_midpoint_score = fixed_midpoint_score
        self._midpoint_score_percentile = midpoint_score_percentile
        self._sharpness = sharpness
        self._novelty_weight = novelty_weight
        self._batch_size = batch_size
        self._should_verify_mutations = should_verify_mutations
        self._fixed_children_per_generation = fixed_children_per_generation
        self._use_process_pool_executors = use_process_pool_executors

        if snapshot_to_resume_from is not None:
            self._load_snapshot(snapshot_to_resume_from)

    def run(self, num_iterations: int) -> Generator[IterationSnapshot, None, None]:
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be positive")

        if not self._evolver:
            self._initialize_evolver()
            # Emit an initial snapshot after the initial organism is evaluated.
            yield self._make_iteration_snapshot(evolver_stats=EvolverStats(num_evaluate_calls=1))

        for _ in range(num_iterations):
            evolver_stats = self._guaranteed_evolver.evolve_iteration(
                self._num_parents_per_iteration, iteration=self._current_iteration
            )
            self._current_iteration += 1

            yield self._make_iteration_snapshot(evolver_stats=evolver_stats)

    def _initialize_evolver(self) -> None:
        assert self._evolver is None, "Evolver has already been initialized"

        # Evaluate the initial organism
        initial_eval = self._problem.evaluator.evaluate(self._problem.initial_organism)

        if self._midpoint_score_percentile is not None:
            midpoint_score_percentile = self._midpoint_score_percentile
        elif self._fixed_midpoint_score is None:
            # Use default value if no midpoint score or percentile is specified
            midpoint_score_percentile = self._DEFAULT_MIDPOINT_SCORE_PERCENTILE
        else:
            midpoint_score_percentile = None

        # Select population class and class-specific configuration
        if self._fixed_children_per_generation is not None:
            population_cls = FixedTreePopulation
            specific_kwargs = {"fixed_children_per_generation": self._fixed_children_per_generation}
        else:
            population_cls = WeightedSamplingPopulation
            specific_kwargs = {
                "fixed_midpoint_score": self._fixed_midpoint_score,
                "midpoint_score_percentile": midpoint_score_percentile,
                "sharpness": self._sharpness if self._sharpness is not None else self._DEFAULT_SHARPNESS,
                "novelty_weight": self._novelty_weight
                if self._novelty_weight is not None
                else self._DEFAULT_NOVELTY_WEIGHT,
            }

        population = population_cls(
            initial_organism=self._problem.initial_organism, initial_evaluation_result=initial_eval, **specific_kwargs
        )

        for mutator in self._problem.mutators:
            mutator.set_context(MutatorContext(population=population))

        self._evolver = Evolver(
            population=population,
            mutators=self._problem.mutators,
            evaluator=self._problem.evaluator,
            learning_log_view_type=self._learning_log_view_type,
            mutator_concurrency=self._mutator_concurrency,
            evaluator_concurrency=self._evaluator_concurrency,
            batch_size=self._batch_size,
            should_verify_mutations=self._should_verify_mutations,
            use_process_pool_executors=self._use_process_pool_executors,
        )

    def _load_snapshot(self, snapshot: bytes) -> None:
        assert self._evolver is None, "Evolver has already been initialized"

        snapshot_dict = pickle.loads(snapshot)
        if not isinstance(snapshot_dict, dict):
            raise ValueError("Snapshot must be a pickled dictionary")

        # Determine which population class to instantiate
        population_snapshot_dict = pickle.loads(snapshot_dict["population_snapshot"])
        class_name = population_snapshot_dict.get("class_name", "WeightedSamplingPopulation")
        if class_name == "FixedTreePopulation":
            population_cls = FixedTreePopulation
        elif class_name == "WeightedSamplingPopulation":
            population_cls = WeightedSamplingPopulation
        else:
            raise ValueError(f"Unrecognized Population class {class_name}")

        population = population_cls.from_snapshot(snapshot_dict["population_snapshot"])
        self._current_iteration = snapshot_dict["current_iteration"]

        for mutator in self._problem.mutators:
            mutator.set_context(MutatorContext(population=population))

        # Overwrite the snapshot's parameters if we have them explicitly specified
        if isinstance(population, WeightedSamplingPopulation):
            if self._sharpness is not None:
                population.set_sharpness(self._sharpness)
            if self._midpoint_score_percentile is not None:
                population.set_midpoint_score_percentile(self._midpoint_score_percentile)
            if self._fixed_midpoint_score is not None:
                population.set_fixed_midpoint_score(self._fixed_midpoint_score)
            if self._novelty_weight is not None:
                population.set_novelty_weight(self._novelty_weight)
        elif isinstance(population, FixedTreePopulation):
            if self._fixed_children_per_generation is not None:
                population._fixed_children_per_generation = self._fixed_children_per_generation

        self._evolver = Evolver(
            population=population,
            mutators=self._problem.mutators,
            evaluator=self._problem.evaluator,
            learning_log_view_type=self._learning_log_view_type,
            mutator_concurrency=self._mutator_concurrency,
            evaluator_concurrency=self._evaluator_concurrency,
            batch_size=self._batch_size,
            should_verify_mutations=self._should_verify_mutations,
            use_process_pool_executors=self._use_process_pool_executors,
        )

    def _snapshot(self) -> bytes:
        population_snapshot = self._guaranteed_evolver.population.snapshot()
        snapshot_dict = {
            "population_snapshot": population_snapshot,
            "current_iteration": self._current_iteration,
        }

        return pickle.dumps(snapshot_dict)

    def _make_iteration_snapshot(self, evolver_stats: EvolverStats) -> IterationSnapshot:
        return IterationSnapshot(
            iteration=self._current_iteration,
            population_size=len(self._guaranteed_evolver.population.organisms),
            snapshot=self._snapshot(),
            population_json_log=self._guaranteed_evolver.population.log_to_json_dict(),
            best_organism_result=self._guaranteed_evolver.population.get_best(),
            score_percentiles=self._guaranteed_evolver.population.get_score_percentiles(),
            evolver_stats=evolver_stats,
        )

    @property
    def _guaranteed_evolver(self) -> Evolver:
        assert self._evolver is not None, "Evolver has not been initialized"
        return self._evolver

    @property
    def population(self) -> Population:
        """Get the current population."""
        if self._evolver is None:
            raise ValueError("Evolver has not been initialized")
        return self._guaranteed_evolver.population
