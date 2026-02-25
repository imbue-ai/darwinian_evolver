import enum
import json
import os
import random
import re
import resource
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from math import floor

import numpy as np
from anthropic import Anthropic
from anthropic.types import Usage as AnthropicUsage
from func_timeout import FunctionTimedOut
from func_timeout import func_timeout
from google import genai
from openai import OpenAI
from openai.types.responses import ResponseUsage
from pydantic import computed_field
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.arc_agi_poetiq import build_feedback
from darwinian_evolver.problems.arc_agi_poetiq import build_prompt
from darwinian_evolver.problems.arc_agi_poetiq import format_problem
from darwinian_evolver.problems.arc_agi_poetiq import make_example
from darwinian_evolver.problems.arc_agi_poetiq import parse_code_from_llm
from darwinian_evolver.problems.arc_agi_poetiq import soft_score

ANTHROPIC_MODEL = "claude-opus-4-6"
ANTHROPIC_THINKING_BUDGET = 16000
ANTHROPIC_MODEL_COSTS = {
    "claude-sonnet-4-5-20250929": {
        "input_per_1m_tokens": 3.0,
        "output_per_1m_tokens": 15.0,
    },
    "claude-opus-4-5-20251101": {
        "input_per_1m_tokens": 5.0,
        "output_per_1m_tokens": 25.0,
    },
    "claude-opus-4-6": {
        "input_per_1m_tokens": 5.0,
        "output_per_1m_tokens": 25.0,
    },
}

GOOGLE_MODEL_HIGH_THINKING = "gemini-3.1-pro-preview"
GOOGLE_MODEL_HIGH_THINKING_ALT = "gemini-3-flash-preview"
GOOGLE_MODEL_LOW_THINKING = "gemini-3-flash-preview"
GOOGLE_MODEL_COSTS = {
    "gemini-3-pro-preview": {
        "prompt_per_1m_tokens": 2.0,
        "cached_per_1m_tokens": 0.2,
        "generated_per_1m_tokens": 12.0,
    },
    "gemini-3-flash-preview": {
        "prompt_per_1m_tokens": 0.5,
        "cached_per_1m_tokens": 0.05,
        "generated_per_1m_tokens": 3.00,
    },
    "gemini-3.1-pro-preview": {
        "prompt_per_1m_tokens": 2.0,
        "cached_per_1m_tokens": 0.2,
        "generated_per_1m_tokens": 12.0,
    },
}

OPENAI_MODEL = "gpt-5.2-2025-12-11"
OPENAI_MODEL_COSTS = {
    "gpt-5.2-2025-12-11": {
        "input_per_1m_tokens": 1.75,
        "cached_per_1m_tokens": 0.175,
        "output_per_1m_tokens": 14.0,
    },
}

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_MODEL_COSTS = {
    # OpenRouter model costs are approximate and depend on provider routing.
    "moonshotai/kimi-k2.5": {
        "input_per_1m_tokens": 0.6,
        "cached_per_1m_tokens": 0.1,
        "output_per_1m_tokens": 3.0,
    },
}

USE_PROVIDER = os.getenv("PROVIDER", "google")


class ThinkingLevel(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Cost tracking state
_COST_TRACKER = 0.0
_COST_TRACKER_MUTEX = threading.Lock()

# Default memory limit for subprocesses (8GB in bytes)
DEFAULT_SUBPROCESS_MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024

HIGHLIGHT_DIFF_THRESHOLD = 0.7


def _track_cost(cost: float) -> None:
    global _COST_TRACKER
    with _COST_TRACKER_MUTEX:
        _COST_TRACKER += cost


def get_current_cost() -> float:
    with _COST_TRACKER_MUTEX:
        return _COST_TRACKER


def set_process_limits(
    memory_limit_bytes: int = DEFAULT_SUBPROCESS_MEMORY_LIMIT_BYTES,
) -> None:
    """Set up memory limits for subprocess.

    Args:
        memory_limit_bytes: Maximum virtual memory in bytes.
    """
    if memory_limit_bytes > 0:
        try:
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Only set if we're trying to set a limit lower than the current hard limit
            # and if the current soft limit is unlimited (-1) or higher than our target
            if hard == resource.RLIM_INFINITY or memory_limit_bytes <= hard:
                if soft == resource.RLIM_INFINITY or soft > memory_limit_bytes:
                    new_hard = min(memory_limit_bytes, hard) if hard != resource.RLIM_INFINITY else memory_limit_bytes
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, new_hard))
        except (ValueError, resource.error, OSError):
            print("Warning: Failed to set process memory limits.")


def _prompt_llm(prompt: str, thinking_level: ThinkingLevel) -> str:
    if USE_PROVIDER == "google":
        return _prompt_llm_google(prompt, thinking_level)
    elif USE_PROVIDER == "google_alt":
        return _prompt_llm_google(prompt, thinking_level, use_alt_model=True)
    elif USE_PROVIDER == "anthropic":
        return _prompt_llm_anthropic(prompt, thinking_level)
    elif USE_PROVIDER == "openai":
        return _prompt_llm_openai(prompt, thinking_level)
    elif USE_PROVIDER == "random_google_openai":
        if np.random.rand() < 0.5 or thinking_level != ThinkingLevel.HIGH:
            return _prompt_llm_google(prompt, thinking_level)
        else:
            return _prompt_llm_openai(prompt, thinking_level)
    elif USE_PROVIDER == "random_google_google_openai":
        r = np.random.rand()
        if r < 0.3 or thinking_level != ThinkingLevel.HIGH:
            return _prompt_llm_google(prompt, thinking_level)
        elif r < 0.6:
            return _prompt_llm_google(prompt, thinking_level, use_alt_model=True)
        else:
            return _prompt_llm_openai(prompt, thinking_level)
    elif USE_PROVIDER == "random_google_google":
        if np.random.rand() < 0.5:
            return _prompt_llm_google(prompt, thinking_level)
        else:
            return _prompt_llm_google(prompt, thinking_level, use_alt_model=True)
    elif USE_PROVIDER == "random_google_anthropic":
        if np.random.rand() < 0.5 or thinking_level != ThinkingLevel.HIGH:
            return _prompt_llm_google(prompt, thinking_level)
        else:
            return _prompt_llm_anthropic(prompt, thinking_level)
    elif USE_PROVIDER == "openrouter":
        return _prompt_llm_openrouter(prompt, thinking_level)
    else:
        raise ValueError(f"Unsupported provider: {USE_PROVIDER}")


def _track_anthropic_costs(model: str, usage: AnthropicUsage) -> None:
    cost = 0.0
    model_costs = ANTHROPIC_MODEL_COSTS[model]
    cost += usage.input_tokens / 1_000_000 * model_costs["input_per_1m_tokens"]
    cost += usage.output_tokens / 1_000_000 * model_costs["output_per_1m_tokens"]
    _track_cost(cost)


@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=10), reraise=True)
def _prompt_llm_anthropic(prompt: str, thinking_level: ThinkingLevel) -> str:
    client = Anthropic()
    # if thinking_level == ThinkingLevel.HIGH:
    #     anthropic_thinking_effort = "high"
    # elif thinking_level == ThinkingLevel.MEDIUM:
    #     anthropic_thinking_effort = "medium"
    # else:
    #     anthropic_thinking_effort = "low"

    # From the Opus 4.6 System Card:
    # >> Because ARC-AGI is a reasoning-intensive benchmark, Opus 4.6 saturates the available
    #  thinking tokens at all effort levels, leading to very similar scores. Nonetheless, at low effort,
    #  the model is able to save on tokens by stopping early for easier problems. <<
    anthropic_thinking_effort = "low"

    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=128000,
            messages=[{"role": "user", "content": prompt}],
            timeout=3600,
            thinking={"type": "disabled" if thinking_level == ThinkingLevel.LOW else "adaptive"},
            output_config={"effort": anthropic_thinking_effort},
        ) as stream:
            response = stream.get_final_message()
    except Exception as e:
        print(f"Anthropic API call failed: {e}")
        raise

    _track_anthropic_costs(ANTHROPIC_MODEL, response.usage)

    for block in response.content:
        if block.type == "text":
            return block.text

    raise ValueError("Response does not contain a text block.")


def _track_google_costs(model: str, usage: genai.types.UsageMetadata) -> None:
    cost = 0.0
    model_costs = GOOGLE_MODEL_COSTS[model]
    uncached_prompt_tokens = 0
    if usage.prompt_token_count:
        uncached_prompt_tokens = usage.prompt_token_count
    if usage.cached_content_token_count:
        uncached_prompt_tokens -= usage.cached_content_token_count
        cost += usage.cached_content_token_count / 1_000_000 * model_costs["cached_per_1m_tokens"]
    cost += uncached_prompt_tokens / 1_000_000 * model_costs["prompt_per_1m_tokens"]
    if usage.candidates_token_count:
        cost += usage.candidates_token_count / 1_000_000 * model_costs["generated_per_1m_tokens"]
    if usage.thoughts_token_count:
        cost += usage.thoughts_token_count / 1_000_000 * model_costs["generated_per_1m_tokens"]
    _track_cost(cost)


ENABLE_GEMINI_CODE_EXECUTION = True


@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=10), reraise=True)
def _prompt_llm_google(prompt: str, thinking_level: ThinkingLevel, use_alt_model: bool = False) -> str:
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        vertexai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true",
    )

    use_high_thinking_model = thinking_level in (ThinkingLevel.HIGH, ThinkingLevel.MEDIUM)
    if use_alt_model:
        model = GOOGLE_MODEL_HIGH_THINKING_ALT if use_high_thinking_model else GOOGLE_MODEL_LOW_THINKING
    else:
        model = GOOGLE_MODEL_HIGH_THINKING if use_high_thinking_model else GOOGLE_MODEL_LOW_THINKING

    if thinking_level == ThinkingLevel.HIGH:
        google_thinking_level = "high"
    elif thinking_level == ThinkingLevel.MEDIUM:
        # Even though it costs more, we use high thinking for the transfer score evaluations as well, as it is of significant importance for selecting
        # the "right" organism. (currently, transfer score is the only place that passes MEDIUM as the thinking level)
        google_thinking_level = "high"
    else:
        google_thinking_level = "low"

    tools = []
    if ENABLE_GEMINI_CODE_EXECUTION and thinking_level == ThinkingLevel.HIGH:
        tools = [genai.types.Tool(code_execution=genai.types.ToolCodeExecution)]

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_level=google_thinking_level),
            http_options=genai.types.HttpOptions(timeout=1800_000),
            tools=tools,
        ),
    )

    if response.usage_metadata:
        _track_google_costs(model, response.usage_metadata)

    response_text = ""
    for parts in response.candidates[0].content.parts:
        if parts.text:
            response_text += parts.text

    return response_text


def _track_openai_costs(model: str, usage: ResponseUsage) -> None:
    cost = 0.0
    model_costs = OPENAI_MODEL_COSTS[model]
    uncached_input_tokens = usage.input_tokens
    cached_tokens = usage.input_tokens_details.cached_tokens
    uncached_input_tokens -= cached_tokens
    cost += uncached_input_tokens / 1_000_000 * model_costs["input_per_1m_tokens"]
    cost += cached_tokens / 1_000_000 * model_costs["cached_per_1m_tokens"]
    cost += usage.output_tokens / 1_000_000 * model_costs["output_per_1m_tokens"]
    _track_cost(cost)


@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=10), reraise=True)
def _prompt_llm_openai(prompt: str, thinking_level: ThinkingLevel) -> str:
    client = OpenAI()

    if thinking_level == ThinkingLevel.HIGH:
        reasoning_effort = "xhigh"
    elif thinking_level == ThinkingLevel.MEDIUM:
        reasoning_effort = "medium"
    else:
        assert thinking_level == ThinkingLevel.LOW
        reasoning_effort = "low"

    response_stream = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": prompt}],
        timeout=3600,
        reasoning={"effort": reasoning_effort},
        stream=True,
    )

    response = None
    for chunk in response_stream:
        if chunk.type == "response.completed":
            response = chunk.response
            break
        elif chunk.type == "response.failed":
            raise RuntimeError(f"OpenAI response failed: {chunk.response.error.message}")

    if response is None:
        raise RuntimeError("Response stream did not contain a completed response.")

    usage = response.usage
    assert usage
    _track_openai_costs(OPENAI_MODEL, usage)

    return response.output_text


def _track_openrouter_costs(model: str, usage: ResponseUsage) -> None:
    cost = 0.0
    model_costs = OPENROUTER_MODEL_COSTS[model]
    uncached_input_tokens = usage.input_tokens
    cached_tokens = usage.input_tokens_details.cached_tokens
    uncached_input_tokens -= cached_tokens
    cost += uncached_input_tokens / 1_000_000 * model_costs["input_per_1m_tokens"]
    cost += cached_tokens / 1_000_000 * model_costs["cached_per_1m_tokens"]
    cost += usage.output_tokens / 1_000_000 * model_costs["output_per_1m_tokens"]
    _track_cost(cost)


@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=10), reraise=True)
def _prompt_llm_openrouter(prompt: str, thinking_level: ThinkingLevel) -> str:
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1")

        response_stream = client.responses.create(
            model=OPENROUTER_MODEL,
            input=[{"role": "user", "content": prompt}],
            timeout=3600,
            extra_body={"reasoning": {"enabled": thinking_level != ThinkingLevel.LOW}},
            stream=True,
        )

        response = None
        for chunk in response_stream:
            if chunk.type == "response.completed":
                response = chunk.response
                break
            elif chunk.type == "response.failed":
                raise RuntimeError(f"OpenRouter response failed: {chunk.response.error.message}")

        if response is None:
            raise RuntimeError("Response stream did not contain a completed response.")

        usage = response.usage
        assert usage
        _track_openrouter_costs(OPENROUTER_MODEL, usage)
    except Exception as e:
        print(f"OpenRouter API call failed: {e}")
        raise

    return response.output_text


class ArcAgiOrganism(Organism):
    code_block: str
    from_explanation: str | None = None


class ArcAgiEvaluationResult(EvaluationResult):
    correctness_score: float
    transfer_score: float
    transfer_subscores: dict[str, float]
    simplicity_score: float
    simplicity_subscores: dict[str, float]
    gt_score: float | None = None

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        return (
            {
                "Correctness Score": self.correctness_score,
                "Transfer Score": self.transfer_score,
                "Simplicity Score": self.simplicity_score,
                "GT Score": self.gt_score if self.gt_score is not None else "N/A",
            }
            | self.simplicity_subscores
            | self.transfer_subscores
        )

    def format_observed_outcome(self, parent_result: EvaluationResult | None) -> str:
        examples_correct = self.correctness_score > 0.999
        if examples_correct:
            outcome = "Solved all examples correctly"
        else:
            outcome = f"Got the examples only {floor(self.correctness_score * 100)}% correct"
        if self.transfer_score < 1.0:
            if examples_correct:
                outcome += ", but"
            else:
                outcome += " and"
            outcome += " failed to generalize to the challenge inputs."
        else:
            outcome += "."

        return outcome


class ArcAgiEvaluationFailureCase(EvaluationFailureCase):
    success: bool
    output: str
    soft_score: float
    error: str | None


def _run_one_inner(
    code_block: str,
    idx: int,
    iin: list[list[int]],
    oout: list[list[int]],
    timeout_secs: int,
) -> ArcAgiEvaluationFailureCase:
    """Helper function to evaluate a code_block on a given input/output pair within its own process."""
    set_process_limits()

    success = False
    soft = 0.0
    err: str | None = None
    try:
        # Execute the organism's code first, populating the scope.
        scope = {}
        func_timeout(timeout_secs, exec, (code_block, scope))
        transform_fn = scope["transform"]

        arr = func_timeout(timeout_secs, lambda: transform_fn(np.array(iin)))

        if not isinstance(arr, np.ndarray):
            err = "The 'transform' function did not return a NumPy array. Returned type: " + str(type(arr))
        if arr.ndim != 2:
            err = f"The 'transform' function did not return a 2D array. Returned array has {arr.ndim} dimensions."
    except FunctionTimedOut:
        err = f"Code execution exceeded time limit of {timeout_secs} seconds."
        arr = np.array([])
    except Exception:
        err = f"Code execution failed.\n{traceback.format_exc()}"
        arr = np.array([])

    if oout:
        truth = np.array(oout)
        success = bool(arr.shape == truth.shape and np.array_equal(arr, truth))
        soft = soft_score(arr, truth)

    return ArcAgiEvaluationFailureCase(
        data_point_id=str(idx),
        success=success,
        output=json.dumps(arr.tolist()),
        soft_score=soft,
        error=err,
    )


class ArcAgiEvaluator(Evaluator[ArcAgiOrganism, ArcAgiEvaluationResult, ArcAgiEvaluationFailureCase]):
    TIMEOUT_SECS = 30  # 30 seconds

    CODE_SIMPLICITY_SCORE_PROMPT = """
You are an expert code reviewer. Your task is to evaluate the simplicity and generality of the provided Python function based on the following criteria:
1. Number of branches: Fewer branches indicate simpler logic.
2. Number of literals and constants in the code: Fewer constants/literal values indicate more general code.
3. How many "color" values are hard-coded with special behavior in the solution? A color is a specific integer field value in the input grid. (can be zero)

Here is the code:
```python
$$code$$
```

Please output three numbers as a JSON object with the following format:
```json
{
    "branch_count": <number_of_branches>,
    "literal_count": <number_of_literals>,
    "color_hardcoded_count": <number_of_hardcoded_colors>
}
```
Do not include any other text outside the JSON object.

Don't overthink it! An approximate count is sufficient.
"""

    TRANSFER_SCORE_PROMPT = """
You are an expert in solving Abstract Reasoning Corpus (ARC) tasks. Your goal is to analyze input-output examples to derive a pattern, and transform any given input grid into the corresponding output grid.

Someone has already identified a pattern and corresponding transformation in an attempt to solve the example input/output pairs for this problem. The pattern is described below.

The person has also applied the suggested transformation to the so called "challenge" inputs. For the challenge inputs, the correct outputs are not known.

Your task is to assess whether the transformation step works for the challenge inputs, and to check whether the person did correctly apply the steps to the challenge inputs.

# PROBLEM
For context, here is the problem to be solved, including the challenge inputs:

$$problem$$


# PATTERN
Here is the pattern and transformation step that someone has previously identified (this might or might not be the correct pattern):
```
$$explanation$$
```

# RESULTS ON CHALLENGE(S)
And here is what the person claims the outputs are for the challenge inputs, based on their transformation step above:

$$feedback$$

# ASSESSMENT
--
Please assess:
1. [applies_to_challenges] Is the identified pattern general enough to apply to the challenge inputs? Or does it need to be extended, generalized or modified to apply correctly?
2. [correctly_applied_pattern] Carefully review the challenge outputs given by the person. Do they correctly apply their transformation steps to the challenge inputs? You might want to manually apply the transformation steps to the challenge inputs yourself and compare the results to see if there are any discrepancies (even minor ones).
3. [consistent_with_examples] As a final sanity check, is the transformation as applied to the challenge inputs consistent with what is illustrated in the example input/output pairs?

Carefully assess these three aspects and then assign a score for each between 0.0 and 1.0. Only assign a score of 1 if you're COMPLETELY confident that the output is correct.
Pay special attention to part 2 (correctly_applied_pattern).

If there are more than one challenge input, verify the above for each challenge separately and report the lowest score among them.

Be VERY RIGOROUS! Even a small discrepancy (e.g. off-by-one error or incorrect order of colors) or sliver of doubt should lead to a reduction in your assessment score.
If you notice any potential discrepancy, or have even the smallest suspicion that there could be an error, reduce the corresponding score.

Common issues to look out for, among others:
* The transformation step contains a trigger condition that is not met for one of the challenge inputs.
* Even though the transformation step would apply to the challenge input, the result provided by the person does not follow the steps.
* A challenge input uses different color assignments than the examples, and hence the transformation is not applied correctly to the challenge input.
* A challenge input exposes a different variation of the pattern than what the transformation step covers.
* The person mixed up the order of a color or shape when producing the challenge output.

Please output three numbers as a JSON object with the following format:
```json
{
    "applies_to_challenges": <score>,
    "correctly_applied_pattern": <score>,
    "consistent_with_examples": <score>
}
```
Do not include any other text outside the JSON object.
"""

    CORRECTNESS_BASELINE_POINT = 0.2
    FULL_CORRECTNESS_BONUS = 0.1

    def __init__(
        self,
        train_in: list[list[list[int]]],
        train_out: list[list[list[int]]],
        test_in: list[list[list[int]]],
        gt_outputs: list | None = None,
    ):
        super().__init__()
        self._train_in = train_in
        self._train_out = train_out
        self._test_in = test_in
        self._gt_outputs = gt_outputs
        self._baseline_similarity = _compute_baseline_similarity(train_in, train_out)

    def evaluate(self, organism: ArcAgiOrganism) -> ArcAgiEvaluationResult:
        """Executes and validates the organism's code."""
        with ProcessPoolExecutor(max_workers=1) as executor:
            train_results: list[ArcAgiEvaluationFailureCase] = []
            for i, (iin, oout) in enumerate(zip(self._train_in, self._train_out, strict=True)):
                train_results.append(self._run_one(organism.code_block, i, iin, oout, executor=executor))

            all_train_solutions_correct = all(tr.success for tr in train_results)

            # Evaluate on test_in as well and store the outputs into the evaluation result as holdout failure cases for later verification
            test_results: list[ArcAgiEvaluationFailureCase] = []
            for i, iin in enumerate(self._test_in):
                gtout = self._gt_outputs[i] if self._gt_outputs is not None and i < len(self._gt_outputs) else []
                test_results.append(self._run_one(organism.code_block, i, iin, gtout, executor=executor))

        # Auxiliary scores for tie breaking
        simplicity_score, simplicity_subscores = self._score_code_simplicity(organism.code_block)
        if all_train_solutions_correct and organism.from_explanation:
            test_transfer_score, test_transfer_subscores = self._score_test_transfer(
                explanation=organism.from_explanation, test_results=test_results
            )
        else:
            # No need to evaluate transfer score when we haven't even solved the training cases yet.
            test_transfer_score = 0.0
            test_transfer_subscores = {}

        soft_correctness_score = float(np.mean([tr.soft_score for tr in train_results]))

        # Rescale the soft correctness score to be CORRECTNESS_BASELINE_POINT at the baseline solution - this makes sure that the population
        # makes full use of the score range available, which improves the effectiveness of the score-based parent sampling.
        rescaled_soft_correctness_score = self.CORRECTNESS_BASELINE_POINT + (1.0 - self.CORRECTNESS_BASELINE_POINT) * (
            soft_correctness_score - self._baseline_similarity
        ) / (1.0 - self._baseline_similarity)
        rescaled_soft_correctness_score = max(0.0, rescaled_soft_correctness_score)

        # We assign an additional score bonus for fully correct solutions, to make sure that those
        # a fully correct but solution with low auxiliary scores is always scored higher than a partially correct one with high auxiliary scores.
        full_correctness_bonus = self.FULL_CORRECTNESS_BONUS if all_train_solutions_correct else 0.0
        correctness_score = (
            1.0 - self.FULL_CORRECTNESS_BONUS
        ) * rescaled_soft_correctness_score + full_correctness_bonus

        score = 0.9 * correctness_score + 0.07 * test_transfer_score + 0.03 * simplicity_score

        # Not used for training, but useful for analysis.
        if self._gt_outputs is not None:
            gt_score = sum(1.0 if tr.success else 0.0 for tr in test_results) / max(len(test_results), 1)
        else:
            gt_score = None

        return ArcAgiEvaluationResult(
            score=score,
            correctness_score=correctness_score,
            transfer_score=test_transfer_score,
            transfer_subscores=test_transfer_subscores,
            simplicity_score=simplicity_score,
            simplicity_subscores=simplicity_subscores,
            trainable_failure_cases=train_results,
            holdout_failure_cases=test_results,
            gt_score=gt_score,
        )

    def verify_mutation(self, organism: ArcAgiOrganism) -> bool:
        """
        We check that the new organism produces a different prediction from its parent on at least one of its inputs.
        """
        if not organism.parent:
            return True

        parent: ArcAgiOrganism = organism.parent

        parent_results: list[ArcAgiEvaluationFailureCase] = []
        child_results: list[ArcAgiEvaluationFailureCase] = []
        with ProcessPoolExecutor(max_workers=1) as executor:
            for i, iin in enumerate(self._train_in + self._test_in):
                parent_results.append(self._run_one(parent.code_block, i, iin, [], executor=executor))
                child_results.append(self._run_one(organism.code_block, i, iin, [], executor=executor))

        for pr, cr in zip(parent_results, child_results, strict=True):
            if pr.output != cr.output:
                return True

        return False

    def _score_code_simplicity(self, code_block: str, retries_remaining: int = 1) -> tuple[float, dict[str, float]]:
        BRANCH_UNIT = 20.0
        LITERAL_UNIT = 20.0
        COLOR_HARDCODED_UNIT = 2.0
        CODE_LENGTH_UNIT = 3000.0

        prompt = build_prompt(self.CODE_SIMPLICITY_SCORE_PROMPT, code=code_block)
        try:
            response_text = _prompt_llm(prompt, thinking_level=ThinkingLevel.LOW)
        except Exception as e:
            if retries_remaining > 0:
                return self._score_code_simplicity(code_block, retries_remaining - 1)
            print(f"Failed to get simplicity scores from LLM: {e}")
            return 0.0, {}

        m = re.search(r".*(\{.*?\})", response_text, re.DOTALL)
        if not m:
            if retries_remaining > 0:
                return self._score_code_simplicity(code_block, retries_remaining - 1)
            print("Failed to find simplicity scores in response")
            return 0.0, {}

        try:
            scores = json.loads(m.group(1))
            branch_count = scores.get("branch_count", 0)
            literal_count = scores.get("literal_count", 0)
            color_hardcoded_count = scores.get("color_hardcoded_count", 0)

            code_length = len(code_block)

            branch_score = 1.0 / (1.0 + branch_count / BRANCH_UNIT)
            literal_score = 1.0 / (1.0 + literal_count / LITERAL_UNIT)
            color_hardcoded_score = 1.0 / (1.0 + color_hardcoded_count / COLOR_HARDCODED_UNIT)
            length_score = 1.0 / (1.0 + code_length / CODE_LENGTH_UNIT)
            # simplicity_score = (branch_score + literal_score + length_score + color_hardcoded_score) / 4.0
            # When looking only at organisms with high transfer scores, the following coefficients seem very mildly correlated with test performance:
            simplicity_score = 0.2 * literal_score + 0.4 * color_hardcoded_score + 0.4 * (1.0 - length_score)
            return simplicity_score, {
                "Branch Score": branch_score,
                "Literal Score": literal_score,
                "Color Hardcoded Score": color_hardcoded_score,
                "Length Score": length_score,
            }
        except Exception as e:
            if retries_remaining > 0:
                return self._score_code_simplicity(code_block, retries_remaining - 1)
            print(f"Failed to parse simplicity scores from response: {e}")
            return 0.0, {}

    def _score_test_transfer(
        self,
        explanation: str,
        test_results: list[ArcAgiEvaluationFailureCase],
        retries_remaining: int = 1,
    ) -> tuple[float, dict[str, float]]:
        if not explanation.strip():
            return 0.0, {}

        example = make_example(self._train_in, self._train_out, self._test_in)
        should_highlight_diff = self._baseline_similarity >= HIGHLIGHT_DIFF_THRESHOLD
        problem_str = format_problem(example, should_highlight_diff=should_highlight_diff)
        feedback_str = build_feedback([], [], [], test_results)[0]
        prompt = build_prompt(
            self.TRANSFER_SCORE_PROMPT,
            explanation=explanation,
            problem=problem_str,
            feedback=feedback_str,
        )
        try:
            response_text = _prompt_llm(prompt, thinking_level=ThinkingLevel.MEDIUM)
        except Exception as e:
            if retries_remaining > 0:
                return self._score_test_transfer(explanation, test_results, retries_remaining - 1)
            print(f"Failed to get transfer score from LLM: {e}")
            return 0.0, {}

        m = re.search(r".*(\{.*?\})", response_text, re.DOTALL)
        if not m:
            if retries_remaining > 0:
                return self._score_test_transfer(explanation, test_results, retries_remaining - 1)
            print("Failed to find transfer score in response")
            return 0.0, {}

        try:
            scores = json.loads(m.group(1))
            applies_to_challenges = min(1.0, max(0.0, scores.get("applies_to_challenges", 0)))
            correctly_applied_pattern = min(1.0, max(0.0, scores.get("correctly_applied_pattern", 0)))
            consistent_with_examples = min(1.0, max(0.0, scores.get("consistent_with_examples", 0)))
            transfer_score_mult = applies_to_challenges * correctly_applied_pattern * consistent_with_examples
            # transfer_score_add = (applies_to_challenges + correctly_applied_pattern + consistent_with_examples) / 3.0
            # transfer_score = 0.5 * transfer_score_mult + 0.5 * transfer_score_add
            transfer_score = transfer_score_mult
            return transfer_score, {
                "Applies to Challenges": applies_to_challenges,
                "Correctly Applied Pattern": correctly_applied_pattern,
                "Consistent with Examples": consistent_with_examples,
            }
        except Exception as e:
            if retries_remaining > 0:
                return self._score_test_transfer(explanation, test_results, retries_remaining - 1)
            print(f"Failed to parse transfer score from response: {e}")
            return 0.0, {}

    def _run_one(
        self, code_block: str, idx: int, iin: list[list[int]], oout: list[list[int]], executor: ProcessPoolExecutor
    ) -> ArcAgiEvaluationFailureCase:
        # Launch evaluation in a subprocess to enforce resource limits.
        future = executor.submit(_run_one_inner, code_block, idx, iin, oout, self.TIMEOUT_SECS)
        return future.result()


class ArcAgiMutator(Mutator[ArcAgiOrganism, ArcAgiEvaluationFailureCase]):
    # Prompt from https://github.com/poetiq-ai/poetiq-arc-agi-solver/blob/0ef451d1f0e4cb24fcbd21732f0cfd21f73c4ee7/arc_agi/prompts.py with some modifications
    PROMPT_TEMPLATE = """You are an expert in solving Abstract Reasoning Corpus (ARC) tasks. Your goal is to analyze input-output examples and create a 'transform' function that correctly transforms any given input grid into the corresponding output grid.

Here's how to approach the problem:

1. Analyze the Examples:
  *   Identify the key objects in the input and output grids (e.g., shapes, lines, regions).
  *   Determine the relationships between these objects (e.g., spatial arrangement, color, size).
  *   Identify the operations that transform the input objects and relationships into the output objects and relationships (e.g., rotation, reflection, color change, object addition/removal).
  *   Consider the grid dimensions, symmetries, and other visual features.
2. Review the existing solution:
  *   You will be given an existing attempt at solving the problem. However, that solution will be incomplete or might even be entirely incorrect.
  *   Carefully analyze the existing solution to understand whether it can be fixed or needs to be replaced by something different.
3. Formulate a Hypothesis BASED ON THE NATURAL LANGUAGE:
  *   Based on your analysis, formulate a natural language description that works consistently across all examples.
  *   **CRITICAL**: Express your solution as NATURAL LANGUAGE INSTRUCTIONS that a human could follow, not as an algorithm.
  *   **AVOID EDGE CASE THINKING**: ARC tasks have a unified solution that works for all examples. Look for the simpler, more general principle that explains ALL examples uniformly.
  *   Make sure that the transformation step can also be applied to the challenge inputs. Sometimes, the challenge inputs will contain additional variations or edge cases that do not appear in the examples.
  *   Express the solution as a sequence of clear transformation steps.
  *   Prioritize simpler steps first.
  *   Consider these types of transformations:
      *   **Object Manipulation:** Moving, rotating, reflecting, resizing, or copying objects.
      *   **Color Changes:** Changing the color of specific objects or regions.
      *   **Spatial Arrangements:** Rearranging the objects in a specific pattern.
      *   **Object Addition/Removal:** Adding or removing objects based on certain criteria.
      *   **Drawing, Filling, and Copying:** Drawing lines, filling in regions with patterns, or copying patterns from a source to a destination.
      *   **Physics-based Transformations:** Simulating gravity, collisions, or other physical interactions.
4. Break Down Into Steps:
  *   Your explanation MUST include a "## Transformation Steps" section.
  *   List the transformation as numbered steps (e.g., "1. Identify all blue regions", "2. For each region...", etc.)
  *   Each step should be concrete and verifiable against the examples.
  *   The steps should read like instructions for a human, not a computer program.
5. Implement the Code:
  *   Write a Python function called `transform(grid: np.ndarray) -> np.ndarray` that implements your transformation step.
  *   Use NumPy for array manipulations. Other standard libraries are also available.
  *   The code should implement the natural language steps you described.
  *   If it is too hard to distill the transformation step into general code, you can hard-code elements of the solutions as a last resort. Make sure to cover all example AND challenge inputs in that case. ONLY do this if you have exhausted all other options.
6. Test and Refine:
  *   Test your code on all examples. If it fails for any example, refine your hypothesis and code.
  *   Also make sure the code will function correctly on the challenge inputs.
7. Output Format:
  *   Your response must have two parts: First, an explanation with transformation steps. Then, a code block.
  *   **Explanation**:
     - Describe the transformation step in natural language
     - Include a "## Transformation Steps" section with numbered steps
     - Briefly mention what you are changing from the previous solution (~1 sentence)
  *   **Code block**: Include the complete Python code for the `transform` function within a single markdown code block (```python). Do not include any `__name__ == "__main__"` block or any code outside the function definition.
  *   Do not add any other reasoning notes besides the explanation and code block.

# PROBLEM TO SOLVE:

Below is a textual representation of the input-output examples and the challenge(s) to be solved.

$$problem$$
"""

    FEEDBACK_PROMPT_TEMPLATE = """
# EXISTING PARTIAL/INCORRECT SOLUTION:

Following is the current, though not yet correct, solution, together with a corresponding evaluation regarding its outputs on the input examples.

Study the current solution and corresponding evaluation and produce a new solution that fixes the parts that it gets wrong.
$$modification_guidance$$

## Explanation
```
$$explanation$$
```

## Code
```python
$$code$$
```

## Evaluation
$$feedback$$

# REMINDER
Make sure to follow the output format specified earlier:
...explanation...
```python
...code...
```
"""

    MODIFICATION_GUIDANCE_NORMAL = " ".join(
        [
            "First see if you can adapt and/or extend the current solution to make it work on the input examples.",
            "Only if there is no simple fix, you might need to come up with a different transformation step that fits the examples better.",
        ]
    )
    MODIFICATION_GUIDANCE_AGGRESSIVE = " ".join(
        [
            "IMPORTANT: You have already tried to fix this solution several times and it hasn't worked!",
            "Please go back to the drawing board and come up with a different transformation step, based on the example input/outputs.",
        ]
    )

    # We don't know if the solution is indeed incorrect or whether it generalizes, but the LLM doesn't need to know that...
    FEEDBACK_PROMPT_TEMPLATE_STREAMLINE = """
# EXISTING INCORRECT SOLUTION:

Following is the current solution. While this solution solves all of the example inputs correctly, it does not yet generalize to the challenge inputs.
Study the current solution and produce a new solution that generalizes to the challenge inputs, while still solving all the examples correctly.

As few guidelines for improving generalization:
1. Carefully study the *challenge* inputs. Do they contain any variations or edge cases that are not covered by the current solution?
2. Make as few assumptions about the input as is possible and needed to match the input-output examples.
3. While we don't know the correct output for the challenge inputs, make sure that the solution applies to those inputs and makes sense for them.

You can first try to adjust the existing solution. Or, if that doesn't work, try an entirely different idea.

## Explanation
```
$$explanation$$
```

## Code
```python
$$code$$
```

## Evaluation
$$feedback$$

# REMINDER
Make sure to follow the output format specified earlier:
...explanation...
```python
...code...
```
"""

    def __init__(
        self,
        train_in: list[list[list[int]]],
        train_out: list[list[list[int]]],
        test_in: list[list[list[int]]],
        aggressive_fraction: float = 0.5,
    ):
        super().__init__()
        self._train_in = train_in
        self._train_out = train_out
        self._test_in = test_in
        self._aggressive_fraction = aggressive_fraction
        self._baseline_similarity = _compute_baseline_similarity(train_in, train_out)

    def mutate(
        self,
        organism: ArcAgiOrganism,
        failure_cases: list[ArcAgiEvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
        retries_remaining: int = 1,
    ) -> list[ArcAgiOrganism]:
        # Get the full failure cases list, including test cases, directly from the population.
        # We won't use any of the ground truth outputs, but we want to know what the code generated for the test inputs.
        eval_result = None
        for o, er in self._context.population.organisms:
            if o is organism:
                eval_result = er
                break
        if eval_result is None:
            print("Could not find evaluation result for organism during mutation.")
            return []

        example = make_example(self._train_in, self._train_out, self._test_in)
        should_highlight_diff = self._baseline_similarity >= HIGHLIGHT_DIFF_THRESHOLD
        problem_str = format_problem(example, should_highlight_diff=should_highlight_diff)
        message = build_prompt(self.PROMPT_TEMPLATE, problem=problem_str)

        feedback_str = build_feedback(
            eval_result.trainable_failure_cases,
            self._train_in,
            self._train_out,
            eval_result.holdout_failure_cases,
        )[0]

        if all(fc.success for fc in eval_result.trainable_failure_cases):
            # We already solve all the training cases. See if we can come up with a simpler / more general solution.
            feedback_message = build_prompt(
                self.FEEDBACK_PROMPT_TEMPLATE_STREAMLINE,
                explanation=organism.from_explanation or "N/A",
                code=organism.code_block,
                feedback=feedback_str,
            )
        else:
            if np.random.rand() < self._aggressive_fraction:
                modification_guidance = self.MODIFICATION_GUIDANCE_AGGRESSIVE
            else:
                modification_guidance = self.MODIFICATION_GUIDANCE_NORMAL
            feedback_message = build_prompt(
                self.FEEDBACK_PROMPT_TEMPLATE,
                explanation=organism.from_explanation or "N/A",
                code=organism.code_block,
                feedback=feedback_str,
                modification_guidance=modification_guidance,
            )

        message += "\n\n" + feedback_message

        if learning_log_entries:
            learning_log_message = _format_learning_log(learning_log_entries)
            message += "\n\n" + learning_log_message

        try:
            response_text = _prompt_llm(message, thinking_level=ThinkingLevel.HIGH)

            new_code_block = parse_code_from_llm(response_text)

            if new_code_block is None:
                if retries_remaining > 0:
                    return self.mutate(
                        organism,
                        failure_cases,
                        learning_log_entries,
                        retries_remaining - 1,
                    )
                print("LLM mutation did not return any code block.")
                return []

            explanation = response_text.split("```python")[0].strip()

            return [
                ArcAgiOrganism(
                    code_block=new_code_block,
                    from_explanation=explanation or "No explanation provided.",
                    from_change_summary=explanation or None,
                )
            ]
        except Exception as e:
            if retries_remaining > 0:
                return self.mutate(organism, failure_cases, learning_log_entries, retries_remaining - 1)
            print(f"LLM mutation failed: {e}")
            return []

    @property
    def supports_batch_mutation(self) -> bool:
        return True


def _compute_baseline_similarity(train_in: list, train_out: list) -> float:
    """Compute average soft score when just copying input to output.

    This helps identify tasks where input and output are very similar,
    which indicates we should highlight the differences.

    Also used for rescaling correctness scores in the evaluator.
    """
    if not train_in or not train_out:
        return 0.0

    similarities = []
    for inp, out in zip(train_in, train_out, strict=False):
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        # soft_score already handles mismatching shapes and returns 0.0
        similarities.append(soft_score(inp_arr, out_arr))

    return float(np.mean(similarities)) if similarities else 0.0


class ArcAgiCrossoverMutator(Mutator[ArcAgiOrganism, ArcAgiEvaluationFailureCase]):
    """Crossover mutator that combines explanations from multiple parent organisms."""

    CROSSOVER_PROMPT_TEMPLATE = """You are an expert in solving Abstract Reasoning Corpus (ARC) tasks by writing Python code. You are given multiple candidate solutions from different attempts at solving the same problem. Your goal is to synthesize insights from these attempts to create an improved solution.

Here's how to approach the problem:

1. Analyze the Examples:
  *   Study the input-output pairs carefully to understand the transformation pattern.
  *   Focus on identifying the transformation in terms of NATURAL LANGUAGE INSTRUCTIONS that a human could follow.
  *   Think about what visual changes occur, not just algorithmic steps.

2. Review the Parent Solutions:
  *   Multiple solution attempts are provided below, each with an explanation and code.
  *   Analyze what each attempt got right and what it got wrong.
  *   Look for common insights across the attempts - these are likely correct.
  *   Identify where the attempts diverge - one may have the right insight while others don't.

3. Formulate Your Transformation Step:
  *   Synthesize a transformation step that combines the best insights from the parents.
  *   **CRITICAL**: Express your solution as NATURAL LANGUAGE INSTRUCTIONS, not as an algorithm.
  *   **AVOID EDGE CASE THINKING**: ARC tasks have a unified solution that works for all examples.
     - Look for the simpler, more general principle that explains ALL examples uniformly.
  *   The solution should be expressible as a sequence of clear transformation steps.

4. Break Down Into Steps:
  *   Your explanation MUST include a "## Transformation Steps" section.
  *   List the transformation as numbered steps (e.g., "1. Identify all blue regions", "2. For each region...", etc.)
  *   Each step should be concrete and verifiable against the examples.
  *   The steps should read like instructions for a human, not a computer program.

5. Implement the Code:
  *   Write a Python function called `transform(grid: np.ndarray) -> np.ndarray` that implements your transformation step.
  *   Use NumPy for array manipulations. Other standard libraries are also available.
  *   The code should implement the natural language steps you described.

6. Output Format:
  *   Your response must have two parts: First, an explanation with transformation steps. Then, a code block.
  *   **Explanation**:
     - Describe the transformation step in natural language
     - Include a "## Transformation Steps" section with numbered steps
     - Briefly mention what you learned from the parent solutions and what you're changing
  *   **Code block**: Include the complete Python code for the `transform` function within a single markdown code block (```python). Do not include any `__name__ == "__main__"` block or any code outside the function definition.

# PROBLEM TO SOLVE:

Below is a textual representation of the input-output examples and the challenge(s) to be solved.

$$problem$$

# PARENT SOLUTIONS:

Below are multiple solution attempts. Learn from their explanations and code:

$$parent_solutions$$

# YOUR TASK:

Synthesize the insights from the parent solutions above to create an improved solution. Remember:
- Express the transformation as natural language instructions, not algorithms
- Avoid edge case thinking - find the ONE unified solution
- Include a "## Transformation Steps" section with numbered steps
- Make sure your code matches your natural language description
"""

    # Setting this to None will use the novelty weight configured on the population
    # (same as for non-crossover parent sampling)
    # We set it to a higher value to encourage selecting more diverse parents for crossover.
    SAMPLING_NOVELTY_WEIGHT: float | None = 1.0

    def __init__(
        self,
        train_in: list[list[list[int]]],
        train_out: list[list[list[int]]],
        test_in: list[list[list[int]]],
        num_parents_per_crossover: int = 3,
        min_population_size: int = 3,
        crossover_frequency: float = 0.25,
    ):
        super().__init__()
        self._train_in = train_in
        self._train_out = train_out
        self._test_in = test_in
        self._num_parents_per_crossover = num_parents_per_crossover
        self._min_population_size = min_population_size
        self._crossover_frequency = crossover_frequency
        self._baseline_similarity = _compute_baseline_similarity(train_in, train_out)

    def mutate(
        self,
        organism: ArcAgiOrganism,
        failure_cases: list[ArcAgiEvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
        retries_remaining: int = 1,
    ) -> list[ArcAgiOrganism]:
        # Decide whether to perform crossover based on frequency
        if random.random() > self._crossover_frequency:
            return []

        assert self._context is not None, "Mutator context not set"
        if len(self._context.population.organisms) < self._min_population_size:
            return []

        try:
            parents_with_results = self._context.population.sample_parents(
                self._num_parents_per_crossover,
                replace=False,
                novelty_weight=self.SAMPLING_NOVELTY_WEIGHT,
            )
        except ValueError as e:
            print(f"Error sampling parents for crossover: {e}")
            return []

        # Format the parent solutions
        parent_solutions_str = self._format_parent_solutions(parents_with_results)

        # Format the problem with diff highlighting if applicable
        example = make_example(self._train_in, self._train_out, self._test_in)
        should_highlight_diff = self._baseline_similarity >= HIGHLIGHT_DIFF_THRESHOLD
        problem_str = format_problem(example, should_highlight_diff=should_highlight_diff)

        # Build the prompt
        prompt = build_prompt(
            self.CROSSOVER_PROMPT_TEMPLATE,
            problem=problem_str,
            parent_solutions=parent_solutions_str,
        )

        try:
            response_text = _prompt_llm(prompt, thinking_level=ThinkingLevel.HIGH)
            new_code_block = parse_code_from_llm(response_text)

            if new_code_block is None:
                if retries_remaining > 0:
                    return self.mutate(
                        organism,
                        failure_cases,
                        learning_log_entries,
                        retries_remaining - 1,
                    )
                print("Crossover mutation did not return a code block.")
                return []

            explanation = response_text.split("```python")[0].strip()

            # Validate that explanation includes transformation steps
            if not self._has_transformation_steps(explanation):
                print("Warning: Crossover explanation missing transformation steps section")

            # Create the new organism
            mutated_organism = ArcAgiOrganism(
                code_block=new_code_block,
                from_explanation=explanation or "No explanation provided.",
                from_change_summary=explanation or None,
            )

            # Set parent relationships
            mutated_organism.parent = parents_with_results[0][0]  # First parent as main
            mutated_organism.additional_parents = [parent for parent, _ in parents_with_results[1:]]

            return [mutated_organism]

        except Exception as e:
            if retries_remaining > 0:
                return self.mutate(organism, failure_cases, learning_log_entries, retries_remaining - 1)
            print(f"Crossover mutation failed: {e}")
            return []

    def _format_parent_solutions(
        self,
        parents_with_results: list[tuple[ArcAgiOrganism, ArcAgiEvaluationResult]],
    ) -> str:
        """Format parent organisms with their explanations and code."""
        parts = []
        for i, (parent, result) in enumerate(parents_with_results, start=1):
            # Get full evaluation feedback
            feedback_str = build_feedback(
                result.trainable_failure_cases,
                self._train_in,
                self._train_out,
                result.holdout_failure_cases,
            )[0]

            part = f"## Parent Solution {i}\n\n"
            part += "**Explanation**:\n```\n"
            part += parent.from_explanation or "No explanation provided"
            part += "\n```\n\n"
            part += "**Code**:\n```python\n"
            part += parent.code_block
            part += "\n```\n\n"
            part += "**Evaluation**:\n"
            part += feedback_str
            part += "\n"

            parts.append(part)

        return "\n".join(parts)

    def _has_transformation_steps(self, explanation: str) -> bool:
        """Check if explanation contains a transformation steps section."""
        # Look for common patterns indicating steps
        patterns = [
            r"##\s*Transformation\s*Steps",
            r"##\s*Steps",
            r"\*\*Transformation\s*Steps\*\*",
            r"\*\*Steps\*\*",
            r"\d+\.\s+",  # Numbered list
        ]

        for pattern in patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                return True

        return False


INITIAL_CODE_BLOCK = """
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # This is a placeholder --- put your solution here.
    return grid
"""


def make_arc_agi_problem(
    task: dict,
    gt_outputs: list | None = None,
    include_crossover: bool = False,
    crossover_frequency: float = 0.25,
    crossover_min_population_size: int = 3,
    use_process_pool_executors: bool = False,
) -> Problem:
    # gt_outputs are not used in the training process, but can be used to visualize correct solutions during manual analysis.
    train = task.get("train", [])
    test = task.get("test", [])
    train_in = [ex["input"] for ex in train]
    train_out = [ex["output"] for ex in train]
    test_in = [ex["input"] for ex in test]

    initial_organism = ArcAgiOrganism(
        code_block=INITIAL_CODE_BLOCK.strip(),
        from_explanation="Initial placeholder solution.",
    )

    # Build mutators list
    mutators: list[Mutator] = [ArcAgiMutator(train_in, train_out, test_in)]

    if include_crossover:
        mutators.append(
            ArcAgiCrossoverMutator(
                train_in,
                train_out,
                test_in,
                crossover_frequency=crossover_frequency,
                min_population_size=crossover_min_population_size,
            )
        )

    return Problem[ArcAgiOrganism, ArcAgiEvaluationResult, ArcAgiEvaluationFailureCase](
        evaluator=ArcAgiEvaluator(train_in, train_out, test_in, gt_outputs=gt_outputs),
        mutators=mutators,
        initial_organism=initial_organism,
    )


def _format_learning_log(entries: list[LearningLogEntry]) -> str:
    if not entries:
        return ""

    formatted_entries = []
    for step, entry in enumerate(entries, start=1):
        log_str = f"## Attempt {step}\n\n"
        log_str += "### Explanation\n"
        log_str += f"```\n{entry.attempted_change}\n```\n\n"
        log_str += "### Observed outcome\n"
        log_str += f"{entry.observed_outcome}"
        formatted_entries.append(log_str)

    return "\n\n".join(
        [
            "# PREVIOUSLY ATTEMPTED TRANSFORMATION STEPS",
            "Below are the transformation steps that have been attempted already, together with an assessment of whether they worked or not. Please make sure to avoid any steps that didn't yield the correct results!",
        ]
        + formatted_entries
    )
