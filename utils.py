from itertools import product
import numpy as np


def can_be_added(element, capset):
    """Check if element can be added to capset without violating cap set property."""
    if element in capset:
        return False
    capset_set = set(capset)
    n_len = len(element)
    for a in capset:
        needed = tuple((3 - (element[i] + a[i]) % 3) % 3 for i in range(n_len))
        if needed in capset_set:
            return False
    return True


# Cap Set Utilities (efficient, no slow brute force)
def get_all_elements(n):
    """Generate all elements in (Z/3Z)^n."""
    return list(product(range(3), repeat=n))


# Solve function (greedy using priority)
def solve(n, priority):
    """Greedy construction of cap set using priority function."""
    elements = get_all_elements(n)
    scores = [priority(el, n) for el in elements]
    sorted_indices = np.argsort(scores, kind="stable")[::-1]
    elements = [elements[idx] for idx in sorted_indices]
    capset = []
    for element in elements:
        if can_be_added(element, capset):
            capset.append(element)
    return capset
