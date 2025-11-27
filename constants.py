import numpy as np
from utils import get_all_elements, can_be_added, solve
from eval import evaluate_capset

# Evaluation environment globals (utils)
EVAL_ENV = {
    "np": np,
    "get_all_elements": get_all_elements,
    "can_be_added": can_be_added,
    "solve": solve,
    "evaluate_capset": evaluate_capset,
}

# Problem inputs (dimensions)
INPUTS = [3, 4, 5, 6, 7, 8]
KNOWN_BESTS = {
    3: 9,
    4: 20,
    5: 45,
    6: 112,
    7: 236,
    8: 512,
}  # Paper's FunSearch result for n=8

# Prompt prefix (skeleton without priority, matching paper Fig.2a)
PROMPT_PREFIX = '''"""Finds large cap sets."""

import numpy as np
# utils available: get_all_elements, can_be_added, solve, evaluate_capset

def main(n):
    """Runs `solve` on `n`-dimensional cap set and evaluates the output."""
    solution = solve(n)
    return evaluate_capset(solution, n)

def solve(n):
    """Builds a cap set of dimension `n` using `priority` function."""
    # Precompute all priority scores.
    elements = get_all_elements(n)
    scores = [priority(el, n) for el in elements]
    # Sort elements according to the scores.
    elements = elements[np.argsort(scores, kind='stable')[::-1]]
    # Build `capset` greedily, using scores for prioritization.
    capset = []
    for element in elements:
        if can_be_added(element, capset):
            capset.append(element)
    return capset
'''

PRIORITY_DOC = (
    '"""Returns the priority with which we want to add `element` to the cap set."""'
)

# Initial trivial program
INITIAL_PRIORITY_CODE = f"""def priority(element: tuple[int, ...], n: int) -> float:
    {PRIORITY_DOC}
    return 0.0"""
