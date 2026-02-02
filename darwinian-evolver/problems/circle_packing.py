import traceback
from contextlib import redirect_stdout
from io import StringIO

import jinja2
import numpy as np
from anthropic import Anthropic
from func_timeout import FunctionTimedOut
from func_timeout import func_timeout

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _prompt_llm(system_prompt: str, prompt: str) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=16384, system=system_prompt, messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# --- Organism and Failure Case Definitions ---


class CirclePackingOrganism(Organism):
    """An organism representing a circle packing solution."""

    code_block: str
    from_failure_diagnosis: str | None = None


class CirclePackingEvaluationResult(EvaluationResult):
    """Represents the result of evaluating a circle packing organism."""

    def format_observed_outcome(self, parent_result: EvaluationResult | None) -> str:
        outcome = super().format_observed_outcome(parent_result, ndigits=5)
        if self.trainable_failure_cases:
            outcome += f' The produced message upon evaluation was: "{self.trainable_failure_cases[0].error_message}"'
        return outcome


class CirclePackingEvaluationFailureCase(EvaluationFailureCase):
    """Represents a failure in the circle packing evaluation."""

    error_message: str
    output: str = ""
    sum_of_radii: float


# --- Evaluator for Circle Packing ---


class CirclePackingEvaluator(
    Evaluator[CirclePackingOrganism, CirclePackingEvaluationResult, CirclePackingEvaluationFailureCase]
):
    """Evaluates a circle packing organism by executing its code."""

    FIXED_RUNNER_CODE = """
import numpy as np
# This part remains fixed (not evolved)
def run_packing():
    \"\"\"Run the circle packing constructor for n=26\"\"\"
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
"""

    TIMEOUT_SECS = 30 * 60  # 30 minutes

    def _validate_packing(
        self, run_output: tuple[np.ndarray, np.ndarray, float], atol=1e-6
    ) -> tuple[bool, str | None]:
        try:
            centers, radii, reported_sum = run_output
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            n_expected = 26
            if centers.shape != (n_expected, 2):
                return False, f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
            if radii.shape != (n_expected,):
                return False, f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
            if np.any(radii < 0):
                return False, f"Negative radii found for circles at indices: {np.where(radii < 0)[0]}"
            if not np.isclose(np.sum(radii), reported_sum, atol=atol):
                return False, f"Sum of radii ({np.sum(radii):.6f}) does not match reported ({reported_sum:.6f})"

            for i in range(n_expected):
                x, y = centers[i]
                r = radii[i]
                if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
                    return False, f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."

            for i in range(n_expected):
                for j in range(i + 1, n_expected):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j] - atol:
                        return (
                            False,
                            f"Circles {i} & {j} overlap. Dist: {dist:.4f}, Sum Radii: {(radii[i] + radii[j]):.4f}",
                        )

            return True, "The circles are placed correctly."
        except Exception as e:
            return False, f"Validation failed with exception: {e}"

    def evaluate(self, organism: CirclePackingOrganism) -> EvaluationResult:
        """Executes and validates the organism's code."""
        scope = {}
        output_str = ""
        try:
            # Execute the organism's code first, populating the scope.
            func_timeout(self.TIMEOUT_SECS, exec, (organism.code_block, scope))
            # Now execute the runner, which can see the functions defined above.
            func_timeout(self.TIMEOUT_SECS, exec, (self.FIXED_RUNNER_CODE, scope))
            run_packing_fn = scope["run_packing"]
            output = StringIO()
            with redirect_stdout(output):
                centers, radii, sum_radii = func_timeout(self.TIMEOUT_SECS, run_packing_fn)
            output_str = output.getvalue()[:10000]  # Limit output size
        except FunctionTimedOut:
            error_msg = f"Code execution exceeded time limit of {self.TIMEOUT_SECS} seconds."
            failure_case = CirclePackingEvaluationFailureCase(
                data_point_id="full_run", error_message=error_msg, sum_of_radii=0.0
            )
            return CirclePackingEvaluationResult(score=0.0, trainable_failure_cases=[failure_case])
        except Exception:
            error_msg = f"Code execution failed.\n{traceback.format_exc()}"
            failure_case = CirclePackingEvaluationFailureCase(
                data_point_id="full_run", error_message=error_msg, sum_of_radii=0.0
            )
            return CirclePackingEvaluationResult(score=0.0, trainable_failure_cases=[failure_case])

        is_valid, msg = self._validate_packing((centers, radii, sum_radii))

        try:
            float_sum_radii = float(sum_radii)
        except (TypeError, ValueError):
            float_sum_radii = 0.0

        if not is_valid:
            failure_case = CirclePackingEvaluationFailureCase(
                data_point_id="full_run",
                error_message=msg,
                sum_of_radii=float_sum_radii,
                output=output_str,
            )
            return CirclePackingEvaluationResult(score=0.0, trainable_failure_cases=[failure_case])

        failure_case = CirclePackingEvaluationFailureCase(
            data_point_id="full_run",
            error_message="Valid but suboptimal configuration.",
            sum_of_radii=float_sum_radii,
            output=output_str,
        )
        return CirclePackingEvaluationResult(score=float_sum_radii, trainable_failure_cases=[failure_case])


# --- Mutator for Circle Packing ---


class ImproveCirclePackingMutator(Mutator[CirclePackingOrganism, CirclePackingEvaluationFailureCase]):
    """Uses an LLM to mutate the circle packing code."""

    # Prompts are based on "ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution" by Lange et al. 2025
    SYSTEM_PROMPT = """You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is about 2.7.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles.
2. A pure hexagonal arrangement may not be optimal due to edge effects.
3. The densest known circle packings often use a hybrid approach.
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters.
5. Consider strategic placement of circles at square corners and edges.
6. Adjusting the pattern to place larger circles at the center and smaller at the edges.
7. The math literature suggests special arrangements for specific values of n.
8. You can use the scipy.optimize package (e.g., SLSQP) to optimize radii and/or center locations given constraints.

You are evolving the code inside the `construct_packing` function. Ensure that all circles are disjoint and lie inside the unit square. Be creative and try to find a new solution better than the best known result. The function must return two numpy arrays: `centers` (shape (26, 2)) and `radii` (shape (26,)).
"""

    IMPROVEMENT_PROMPT_TEMPLATE = """
The current Python code for the `construct_packing` function is:
```python
{{organism_code_block}}
```

The sum of the radii in this configuration is {{sum_of_radii}}. The best known result is about 2.7.

When this code was evaluated, it produced the following error (or `None` if it was valid but suboptimal):
```
{{failure_error_message}}
```

This was its output when executed:
```
{{failure_output}}
```

Please provide an improved version of the construct_packing function to achieve a valid configuration with a higher sum of radii.

Feel free to include print statements in the new function to help understand its behavior for further improvement.

Your response should consist of the following three parts:
1. Start with a plan of how you're thinking to improve the function
2. Then provide the Python code for the new function, enclosed in a single markdown code block (enclosed in triple backticks). The code must define construct_packing() and any helper functions it needs.
3. Finally, right after the closing triple backticks, provide a short (2-3 sentence) summary of the change you've made to the code. Be specific but concise. Do not add any headings or formatting markers, just a paragraph of text. Then end your response right afterwards.

{% if learning_log_entries %}For guidance, consider the following changes that have been tried, some successfully and some unsuccessfully:
{% for learning_log_entry in learning_log_entries %}
=== Attempted Change {{ loop.index }} Start ===
{{ learning_log_entry.attempted_change }}
--- Observed Outcome ---
{{ learning_log_entry.observed_outcome }}
=== Attempted Change {{ loop.index }} End ===
{% endfor %}{% endif %}
"""

    def mutate(
        self,
        organism: CirclePackingOrganism,
        failure_cases: list[CirclePackingEvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[CirclePackingOrganism]:
        if not failure_cases:
            return []

        failure_case = failure_cases[0]

        user_prompt = (
            jinja2.Template(self.IMPROVEMENT_PROMPT_TEMPLATE)
            .render(
                organism_code_block=organism.code_block,
                sum_of_radii=failure_case.sum_of_radii,
                failure_error_message=failure_case.error_message or "None",
                failure_output=failure_case.output,
                learning_log_entries=learning_log_entries,
            )
            .strip()
        )

        try:
            response_text = _prompt_llm(self.SYSTEM_PROMPT, user_prompt)

            parts = response_text.split("```")
            if len(parts) != 3:
                print("LLM mutation did not return the expected format.")
                return []

            diagnosis, new_code_block, change_summary = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if new_code_block.startswith("python"):
                new_code_block = new_code_block[6:].strip()

            return [
                CirclePackingOrganism(
                    code_block=new_code_block,
                    from_failure_diagnosis=diagnosis,
                    from_change_summary=change_summary,
                )
            ]
        except Exception as e:
            print(f"LLM mutation failed: {e}")
            return []


INITIAL_CODE_BLOCK = '''
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)]

    # Place 17 more circles in an outer ring
    for i in range(17):
        angle = 2 * np.pi * i / 17
        centers[i + 9] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]

    radii = np.full(n, 0.05)
    return centers, radii
'''


def make_circle_packing_problem() -> Problem:
    """Creates the circle packing problem instance."""
    initial_organism = CirclePackingOrganism(code_block=INITIAL_CODE_BLOCK.strip())
    return Problem[CirclePackingOrganism, CirclePackingEvaluationResult, CirclePackingEvaluationFailureCase](
        evaluator=CirclePackingEvaluator(),
        mutators=[ImproveCirclePackingMutator()],
        initial_organism=initial_organism,
    )
