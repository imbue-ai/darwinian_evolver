"""
An example problem where the goal is to evolve a prompt for verifying multiplication results.

This is a slightly more advanced example, which includes support for batch mutation and post-mutation verification.
"""

from __future__ import annotations

import enum
from random import Random
from uuid import uuid4

import jinja2
from anthropic import Anthropic
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem

EVAL_MODEL = "claude-3-5-haiku-20241022"
MUTATION_MODEL = "claude-sonnet-4-20250514"


def _prompt_llm(prompt: str, model_name: str) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model_name, max_tokens=8000, messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


class ResponseParseError(Exception):
    raw_response: str

    def __init__(self, raw_response: str, message: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class MultiplicationVerifierOrganism(Organism):
    prompt_template: str

    # Additional field for inspecting the mutation process in the lineage visualizer.
    # This is not necessary for the problem to work, but can be useful for debugging.
    from_failure_diagnosis: str | None = None

    def run(self, a: int, b: int, result: int) -> tuple[str, bool]:
        """Evaluate the organism on a given multiplication verification task."""
        prompt = jinja2.Template(self.prompt_template.strip()).render(a=a, b=b, result=result)

        raw_response = _prompt_llm(prompt, EVAL_MODEL)

        if raw_response.strip().lower().endswith("true"):
            boolean_response = True
        elif raw_response.strip().lower().endswith("false"):
            boolean_response = False
        else:
            raise ResponseParseError(raw_response, "Response does not end with 'true' or 'false'.")

        return raw_response, boolean_response


class MultiplicationVerifierDataPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))

    a: int
    b: int
    result: int
    correct_result: int

    @computed_field
    def is_correct(self) -> bool:
        return self.result == self.correct_result


class MultiplicationVerifierEvaluationFailureCase(EvaluationFailureCase):
    data_point: MultiplicationVerifierDataPoint

    full_llm_response: str
    boolean_llm_response: bool | None
    error_message: str | None = None


class FailureTypes(enum.Enum):
    PARSE_ERROR = "parse_error"
    TEMPLATE_ERROR = "template_error"
    FALSE_ACCEPT = "false_accept"
    FALSE_REJECT = "false_reject"


class ImproveMultiplicationVerifierMutator(
    Mutator[MultiplicationVerifierOrganism, MultiplicationVerifierEvaluationFailureCase]
):
    IMPROVEMENT_PROMPT_TEMPLATE = """
We want to build a prompt that allows an LLM (large language model) to verify the result of a multiplication between two numbers.
Given two numbers a and b, and a result, the LLM should determine if the result is the correct result of multiplying a and b.
As an additional constraint, we know that if the result is incorrect, it will be incorrect in only one digit.

The current prompt template is:
```
{{ organism.prompt_template }}
```

The prompt template is converted to a prompt by replacing the placeholders, a, b, and result respectively with the actual numbers. It is then sent to an LLM.
The LLM's response can start with arbitrary text, but must end with either the word true or false, indicating whether the multiplication result is correct.

Unfortunately, we found cases where the LLM made incorrect determinations.

Here are the details of these cases:
{% for failure_case in failure_cases %}
=== Failure Case {{ loop.index }} Start ===
The inputs were:
a = {{ failure_case.data_point.a }}
b = {{ failure_case.data_point.b }}
result = {{ failure_case.data_point.result }}

The LLM's response was:
```
{{ failure_case.full_llm_response }}
```

{% if failure_case.error_message is not none %}There was an error in processing the LLM's response: {{ failure_case.error_message }}{% endif %}
{% if failure_case.boolean_llm_response is not none and failure_case.boolean_llm_response %}The LLM determined that the result was correct, but actually it was incorrect. The correct multiplication result would have been {{ failure_case.data_point.correct_result }}.{% endif %}
{% if failure_case.boolean_llm_response is not none and not failure_case.boolean_llm_response %}The LLM determined that the result was incorrect, but it was actually correct.{% endif %}
=== Failure Case {{ loop.index }} End ===
{% endfor %}

Please:
1. First diagnose what went wrong in each failure case.
2. Then, come up with a potential improvement for the multiplication verification prompt.
3. Suggest an updated prompt template. Wrap the improved prompt template in triple backticks.
4. Finally, right after the closing triple backticks, provide a short (2-3 sentence) summary of the change you've made to the prompt template. Be specific but concise. Do not add any headings or formatting markers, just a paragraph of text. Then end your response right afterwards.

Do not under any circumstance include the specific numbers from the example in the improved prompt template. Rather, think of general ways to improve the prompt.
{% if is_single_step %}
If you're adding a new technique or instruction to the prompt template, *only add a single one*. Try to pick one that is directly relevant to the failure cases.
{% endif %}

{% if learning_log_entries %}For guidance, consider the following changes that have been tried, some successfully and some unsuccessfully:
{% for learning_log_entry in learning_log_entries %}
=== Attempted Change {{ loop.index }} Start ===
{{ learning_log_entry.attempted_change }}
--- Observed Outcome ---
{{ learning_log_entry.observed_outcome }}
=== Attempted Change {{ loop.index }} End ===
{% endfor %}{% endif %}
"""

    _is_single_step: bool

    def __init__(self, is_single_step: bool = True) -> None:
        """
        Initialize the mutator.

        is_single_step attempts to reduce the number of changes made to the prompt template in each iteration.
        This intentionally weakens the mutator, but makes it more representative of what you might see in complex real-world problems.
        """
        self._is_single_step = is_single_step

    def mutate(
        self,
        organism: MultiplicationVerifierOrganism,
        failure_cases: list[MultiplicationVerifierEvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[MultiplicationVerifierOrganism]:
        prompt = (
            jinja2.Template(self.IMPROVEMENT_PROMPT_TEMPLATE.strip())
            .render(
                organism=organism,
                failure_cases=failure_cases,
                is_single_step=self._is_single_step,
                learning_log_entries=learning_log_entries,
            )
            .strip()
        )

        improvement_response = _prompt_llm(prompt, MUTATION_MODEL)

        try:
            failure_diagnosis, improved_prompt_template, change_summary = self._parse_response(improvement_response)
        except ValueError as e:
            print(f"Error parsing improvement response: {e}")
            return []

        return [
            MultiplicationVerifierOrganism(
                prompt_template=improved_prompt_template,
                from_failure_diagnosis=failure_diagnosis,
                from_change_summary=change_summary,
            ),
        ]

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """Parse the response from the LLM to extract 1. The failure diagnosis, and 2. The improved prompt template."""
        # Find the last ```-enclosed block in the response
        parts = response.split("```")
        if len(parts) != 3:
            raise ValueError("Response does not contain the expected sections.")

        # The parts should be:
        failure_diagnosis, improved_prompt_template, change_summary = (
            parts[0].strip(),
            parts[1].strip(),
            parts[2].strip(),
        )

        return failure_diagnosis, improved_prompt_template, change_summary


class MultiplicationVerifierEvaluator(
    Evaluator[MultiplicationVerifierOrganism, EvaluationResult, MultiplicationVerifierEvaluationFailureCase]
):
    _MAX_RANGE = 1000000
    _NUM_TRAINABLE_DATAPOINTS = 15
    _NUM_HOLDOUT_DATAPOINTS = 5

    def __init__(self) -> None:
        self._trainable_data = []
        self._holdout_data = []
        rnd = Random(42)
        for _ in range(self._NUM_TRAINABLE_DATAPOINTS):
            self._trainable_data.append(self._make_data_point(rnd))
        for _ in range(self._NUM_HOLDOUT_DATAPOINTS):
            self._holdout_data.append(self._make_data_point(rnd))

    def evaluate(self, organism: MultiplicationVerifierOrganism) -> EvaluationResult:
        trainable_failure_cases = []
        holdout_failure_cases = []
        for data_point in self._trainable_data:
            maybe_failure_case = self._evaluate_data_point(organism, data_point)
            if maybe_failure_case is not None:
                trainable_failure_cases.append(maybe_failure_case)
        for data_point in self._holdout_data:
            maybe_failure_case = self._evaluate_data_point(organism, data_point)
            if maybe_failure_case is not None:
                holdout_failure_cases.append(maybe_failure_case)

        # In this evaluator, we use both trainable and holdout phrases to calculate a success score.
        # Depending on your requirements, you could decide to use only the holdout phrases for scoring.
        num_total = len(self._trainable_data) + len(self._holdout_data)
        num_correct = num_total - len(trainable_failure_cases) - len(holdout_failure_cases)
        score = num_correct / num_total

        return EvaluationResult(
            score=score,
            trainable_failure_cases=trainable_failure_cases,
            holdout_failure_cases=holdout_failure_cases,
        )

    def verify_mutation(self, organism: MultiplicationVerifierOrganism) -> bool:
        """Verify that the mutation of the organism has addressed at least one of the given failure cases."""
        failure_cases = organism.from_failure_cases
        assert failure_cases is not None
        for failure_case in failure_cases:
            data_point = failure_case.data_point
            maybe_failure_case = self._evaluate_data_point(organism, data_point)
            if maybe_failure_case is None:
                # If the mutation did no longer produce a failure case for this data point, we consider it a success.
                # Note that this type of verification is not entirely reliable, since some failures can originate from
                # randomness in the LLM response and might fail/pass in a given run just by chance.
                return True

        return False

    def _evaluate_data_point(
        self,
        organism: MultiplicationVerifierOrganism,
        data_point: MultiplicationVerifierDataPoint,
    ) -> MultiplicationVerifierEvaluationFailureCase | None:
        try:
            full_llm_response, boolean_llm_response = organism.run(data_point.a, data_point.b, data_point.result)
        except ResponseParseError as e:
            return MultiplicationVerifierEvaluationFailureCase(
                data_point_id=data_point.id,
                failure_type=FailureTypes.PARSE_ERROR.value,
                data_point=data_point,
                full_llm_response=e.raw_response,
                boolean_llm_response=None,
                error_message=str(e),
            )
        except jinja2.TemplateError as e:
            return MultiplicationVerifierEvaluationFailureCase(
                data_point_id=data_point.id,
                failure_type=FailureTypes.TEMPLATE_ERROR.value,
                data_point=data_point,
                full_llm_response="",
                boolean_llm_response=None,
                error_message=f"Error in template rendering: {str(e)}",
            )

        if boolean_llm_response != (data_point.is_correct):
            failure_type = FailureTypes.FALSE_ACCEPT.value if boolean_llm_response else FailureTypes.FALSE_REJECT.value
            return MultiplicationVerifierEvaluationFailureCase(
                data_point_id=data_point.id,
                failure_type=failure_type,
                data_point=data_point,
                full_llm_response=full_llm_response,
                boolean_llm_response=boolean_llm_response,
            )
        else:
            return None

    def _make_data_point(self, rnd: Random) -> MultiplicationVerifierDataPoint:
        a = rnd.randint(1, self._MAX_RANGE)
        b = rnd.randint(1, self._MAX_RANGE)
        correct_result = a * b
        if rnd.random() < 0.5:
            result = self._replace_random_digit(correct_result, rnd)
            assert result != correct_result
        else:
            result = correct_result

        return MultiplicationVerifierDataPoint(a=a, b=b, result=result, correct_result=correct_result)

    def _replace_random_digit(self, number: int, rnd: Random) -> int:
        """Replace a single digit in the number."""
        digits = list(str(number))
        index_to_replace = rnd.randint(0, len(digits) - 1)
        original_digit = digits[index_to_replace]
        new_digit = original_digit
        while new_digit == original_digit:
            new_digit = str(rnd.randint(0, 9))
        digits[index_to_replace] = new_digit
        return int("".join(digits))


INITIAL_PROMPT_TEMPLATE = """
Your task is to verify the result of a multiplication between two numbers.

{{ a }} * {{ b }} = {{ result }}

Is this result correct? Finish your answer with either true or false, nothing after that.
"""


def make_multiplication_verifier_problem() -> Problem:
    initial_organism = MultiplicationVerifierOrganism(prompt_template=INITIAL_PROMPT_TEMPLATE.strip())

    return Problem[MultiplicationVerifierOrganism, EvaluationResult, MultiplicationVerifierEvaluationFailureCase](
        evaluator=MultiplicationVerifierEvaluator(),
        mutators=[ImproveMultiplicationVerifierMutator()],
        initial_organism=initial_organism,
    )
