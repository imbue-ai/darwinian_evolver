# ---- Code in this file is taken from https://github.com/poetiq-ai/poetiq-arc-agi-solver/, with some small modifications applied.
#
# Copyright 2025 Poetiq, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import re
from math import floor
from typing import Any

import numpy as np

USE_SLASH_DIFF = True


def build_prompt(base_prompt: str, **fields: str) -> str:
    s = base_prompt
    for k, v in fields.items():
        s = s.replace(f"$${k}$$", v)
    return s


def _array_diff(arr1: np.ndarray, arr2: np.ndarray) -> str:
    rows, cols = arr1.shape
    out = []
    has_any_diff = np.any(arr1 != arr2)
    for i in range(rows):
        row = []
        for j in range(cols):
            if arr1[i, j] == arr2[i, j]:
                s = str(int(arr1[i, j]))
                if has_any_diff:
                    s = "  " + s
                row.append(s)
            else:
                assert has_any_diff
                row.append(f"{int(arr1[i, j])}/{int(arr2[i, j])}")
        out.append(" ".join(row))
    return "\n".join(out)


def parse_code_from_llm(response: str) -> str | None:
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def soft_score(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        return 0.0
    if truth.size == 0:
        return 1.0
    raw = np.mean(pred == truth)
    return float(np.nan_to_num(raw, posinf=0.0, neginf=0.0))


def make_example(train_in, train_out, test_in) -> dict[str, Any]:
    train = [{"input": iin, "output": oout} for iin, oout in zip(train_in, train_out, strict=True)]
    test = [{"input": iin} for iin in test_in]
    return {"train": train, "test": test}


def format_problem(problem: dict[str, Any], should_highlight_diff: bool = False) -> str:
    """Format problem description, optionally highlighting diffs for high-similarity tasks.

    Args:
        problem: The problem dictionary containing train and test examples
        should_highlight_diff: If True, show changed cells in [brackets] and unchanged cells as _
    """
    train = list(problem["train"])
    test = list(problem["test"])

    example_str = ""
    challenge_str = ""

    for example_num, example in enumerate(train, start=1):
        example_str += f"""
## Example #{example_num}

### Input:
```
{example_to_diagram(example["input"])}
```

"""

        example_str += f"""### Output:
```
{example_to_diagram(example["output"])}
```
"""
        if should_highlight_diff:
            # Inline diff highlighting logic
            inp_arr = np.array(example["input"])
            out_arr = np.array(example["output"])

            if inp_arr.shape == out_arr.shape:
                # Same shape - highlight diffs
                if USE_SLASH_DIFF:
                    example_str += f"""
### Input/Output Comparison:
For cells that are different between input and output, the format is 'input/output'. Cells that are the same are shown as-is.
```
{_array_diff(inp_arr, out_arr)}
```
"""
                else:
                    example_str += f"""
### Output (differences from input only):
```
{example_to_diff_diagram(inp_arr, out_arr)}
```
"""

    for challenge_num, challenge in enumerate(test, start=1):
        challenge_str += f"""
## Challenge #{challenge_num}

### Input:
```
{example_to_diagram(challenge["input"])}
```
"""

    return example_str + challenge_str


def example_to_diagram(example: list[list[int]] | np.ndarray) -> str:
    """Converts an ARC-AGI example (list of lists) to a diagram (ascii grid)."""
    diagram = ""
    for row in example:
        row_str = " ".join([str(col) for col in row]) + "\n"
        diagram += row_str
    return diagram[:-1]  # Strip final \n


def example_to_diff_diagram(
    example_in: list[list[int]] | np.ndarray, example_out: list[list[int]] | np.ndarray
) -> str:
    """Converts an ARC-AGI example (list of lists) to a diagram (ascii grid)."""
    diagram = ""
    for i, row in enumerate(example_out):
        row_str = " ".join([f"{col}" if col != example_in[i][j] else "-" for j, col in enumerate(row)]) + "\n"
        diagram += row_str
    return diagram[:-1]  # Strip final \n


def _parse_json_array_no_expand(s: str) -> np.ndarray | None:
    """Parse JSON into a NumPy array without changing rank or dtype."""
    try:
        return np.array(json.loads(s))
    except Exception:
        return None


def build_feedback(
    train_results: list["ArcAgiEvaluationFailureCase"],
    train_in,
    train_out,
    test_results=None,
) -> tuple[str, float]:
    feedback_parts: list[str] = []
    per_example_scores: list[float] = []

    train_results = sorted(train_results, key=lambda fc: int(fc.data_point_id))

    for fc in train_results:
        i = int(fc.data_point_id)
        if fc.success:
            feedback_parts.append(f"Solves Example #{i + 1} correctly. ")
            per_example_scores.append(1.0)
            continue

        msg_lines: list[str] = [f"Solves Example #{i + 1} INCORRECTLY. "]

        pred_raw = _parse_json_array_no_expand(fc.output) if fc.output else None
        truth = np.array(train_out[i])

        if pred_raw is None:
            per_example_scores.append(0.0)
            msg_lines.append("\nThe output has to be a rectangular grid of numbers.\n")
        else:
            pred_for_display = pred_raw
            if pred_for_display.ndim < 2:
                pred_for_display = np.expand_dims(pred_for_display, axis=list(range(2 - pred_for_display.ndim)))

            if pred_raw.shape != truth.shape:
                per_example_scores.append(0.0)
                msg_lines.append(
                    f"\n\nShape mismatch: your prediction's shape was {pred_raw.shape}, while the correct shape was {truth.shape}."
                )
            else:
                # Same shape: show diff grid and compute soft score.
                msg_lines.append(
                    "".join(
                        [
                            "\nYour code's output does not match the expected output.",
                            "\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n",
                            "Correctly predicted values are shown as-is while the incorrectly predicted values are shown ",
                            "in the format 'prediction/correct':\n",
                        ]
                    )
                )
                diff = _array_diff(pred_for_display, truth)
                msg_lines.append(f"\n```\n{diff}\n```\n")

                example_score = float(np.mean(pred_raw == truth))
                example_score = float(np.nan_to_num(example_score, posinf=0.0, neginf=0.0))
                per_example_scores.append(example_score)
                rounded_example_score = floor(example_score * 100) / 100.0
                msg_lines.append(
                    f"Output accuracy: {rounded_example_score} (0 is worst, 1 is best. The goal is perfect accuracy!).\n"
                )

        if fc.error:
            msg_lines.append(f"\n\nYour code produced the following error:\n{fc.error}\n")

        feedback_parts.append("".join(msg_lines))

    # Extension: Also provide the outputs on the test challenges, even though we don't know if they are correct.
    for fc in test_results or []:
        i = int(fc.data_point_id)
        msg_lines: list[str] = [f"Challenge #{i + 1} output:"]

        pred_raw = _parse_json_array_no_expand(fc.output) if fc.output else None

        if pred_raw is None:
            msg_lines.append("\nThe output was invalid - it has to be a rectangular grid of numbers.\n")
        else:
            pred_for_display = pred_raw
            if pred_for_display is not None and pred_for_display.ndim < 2:
                pred_for_display = np.expand_dims(pred_for_display, axis=list(range(2 - pred_for_display.ndim)))

            msg_lines.append(
                "\nThe code produced the following prediction. We do not know whether this prediction is correct or not:\n"
            )
            diff = _array_diff(pred_for_display, pred_for_display)
            msg_lines.append(f"\n```\n{diff}\n```\n")

        feedback_parts.append("".join(msg_lines))

        if fc.error:
            msg_lines.append(f"\n\nYour code produced the following error:\n{fc.error}\n")

    full_feedback = "\n\n".join(feedback_parts)
    mean_score = (
        float(np.mean(np.nan_to_num(per_example_scores, posinf=0.0, neginf=0.0))) if per_example_scores else 0.0
    )
    return full_feedback, mean_score


# Functions from run_arc.py


def grids_equal(a, b) -> bool:
    """Strict structural equality for ARC grids (list[list[int]])."""
    return a == b


def score_task(kaggle_preds: list[dict], gt_outputs: list) -> float:
    """
    Fraction of test inputs correct for a task.
    Correct if attempt_1 == GT or attempt_2 == GT for each test input.
    """
    if not gt_outputs:
        return 0.0
    correct = 0
    for i, gt in enumerate(gt_outputs):
        if i >= len(kaggle_preds):
            continue
        pack = kaggle_preds[i] or {}
        a1 = pack.get("attempt_1")
        a2 = pack.get("attempt_2")
        if (a1 is not None and grids_equal(a1, gt)) or (a2 is not None and grids_equal(a2, gt)):
            correct += 1
    return correct / max(len(gt_outputs), 1)


def coerce_grid(x: Any) -> list:
    # numpy -> list
    try:
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # stringified JSON -> list
    if isinstance(x, str):
        s = x.strip()
        if s and (s[0] == "[" or s[0] == "{"):
            try:
                parsed = json.loads(s)
                return parsed
            except Exception:
                # not JSON; fall through
                return []
        else:
            return []
    # already list-like?
    if isinstance(x, list):
        return x
    return []


def build_kaggle_two_attempts(results: list[list["ArcAgiEvaluationFailureCase"]], test_in: list[list[list[int]]]):
    """
    Returns: List[{"attempt_1": grid, "attempt_2": grid}] with len == len(test_in).
    """
    num_tests = len(test_in)
    out = []

    for j in range(num_tests):
        attempts: list[list] = []

        # Sweep iterations in order; collect up to 2 successful outputs for test j
        for tr in results:
            tr = sorted(tr, key=lambda fc: fc.data_point_id)
            if j < len(tr):
                fc = tr[j]
                grid = coerce_grid(fc.output)
                if grid != []:
                    attempts.append(grid)
                    if len(attempts) == 2:
                        break

        # Pad with empty arrays if fewer than two attempts available
        while len(attempts) < 2:
            attempts.append([])

        out.append({"attempt_1": attempts[0], "attempt_2": attempts[1]})

    return out
