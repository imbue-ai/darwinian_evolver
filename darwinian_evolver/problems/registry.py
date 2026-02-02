from typing import Callable

from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.circle_packing import make_circle_packing_problem
from darwinian_evolver.problems.multiplication_verifier import make_multiplication_verifier_problem
from darwinian_evolver.problems.parrot import make_parrot_problem

AVAILABLE_PROBLEMS: dict[str, Callable[[], Problem]] = {
    "parrot": make_parrot_problem,
    "circle_packing": make_circle_packing_problem,
    "multiplication_verifier": make_multiplication_verifier_problem,
}
