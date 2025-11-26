import numpy as np
import inspect
import re
import random
from typing import List, Tuple, Dict, Optional, Callable
from itertools import product
from collections import defaultdict
# import openai  # Uncomment and pip install openai for real LLM


# Cap Set Utilities (efficient, no slow brute force)
def get_all_elements(n: int) -> List[Tuple[int, ...]]:
    """Generate all elements in (Z/3Z)^n."""
    return list(product(range(3), repeat=n))


def can_be_added(element: Tuple[int, ...], capset: List[Tuple[int, ...]]) -> bool:
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


# No need for slow is_capset brute force: greedy ensures validity


# Solve function (greedy using priority)
def solve(n: int, priority: Callable) -> List[Tuple[int, ...]]:
    """Greedy construction of cap set using priority function."""
    elements = get_all_elements(n)
    scores = [priority(el, n) for el in elements]
    sorted_indices = np.argsort(scores, kind="stable")[::-1]
    elements = [elements[idx] for idx in sorted_indices]
    capset: List[Tuple[int, ...]] = []
    for element in elements:
        if can_be_added(element, capset):
            capset.append(element)
    return capset


def evaluate_capset(solution: List[Tuple[int, ...]], n: int) -> int:
    """Evaluate: return size (always valid due to greedy + can_be_added)."""
    return len(solution)


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
INITIAL_PRIORITY_CODE = (
    """def priority(element: tuple[int, ...], n: int) -> float:
    """
    + PRIORITY_DOC
    + """
    return 0.0"""
)


class ProgramsDB:
    def __init__(self, num_islands: int = 20, max_island_size: int = 2000):
        self.num_islands = num_islands
        self.islands: List[List[Tuple[str, float, tuple]]] = [
            [] for _ in range(num_islands)
        ]
        self.max_island_size = max_island_size
        self.total_evaluations = 0
        self.best_overall: Optional[Tuple[str, float, tuple]] = None
        self.inputs = INPUTS
        self.T_cluster = (
            0.25  # Temperature for cluster selection (lower = more elitist)
        )
        self.T_program = (
            0.05  # Temperature for program length bias (lower = prefer shorter)
        )
        self.reset_every = 512  # Evaluations between island resets
        # Initialize all islands with initial program
        init_score, init_sig = self._evaluate_program(INITIAL_PRIORITY_CODE)
        init_entry = (INITIAL_PRIORITY_CODE, init_score, init_sig)
        for island in self.islands:
            island.append(init_entry)
        self.best_overall = init_entry
        print(
            f"Initialized with initial score: {init_score:.2f} (mean size), signature: {[int(s) for s in init_sig]}"
        )

    def _evaluate_program(
        self, prog_code: str
    ) -> Tuple[Optional[float], Optional[tuple]]:
        """Safely evaluate program on all inputs: return (mean score, signature) or None."""
        try:
            env = EVAL_ENV.copy()
            exec(prog_code, env)
            if "priority" not in env:
                return None, None
            priority_func = env["priority"]
            scores = []
            for n in self.inputs:
                solution = solve(n, priority_func)
                size = evaluate_capset(solution, n)
                scores.append(float(size))
            mean_score = np.mean(scores)
            signature = tuple(scores)
            return mean_score, signature
        except Exception:
            return None, None

    def sample_k_from_island(
        self, island: List[Tuple[str, float, tuple]], k: int = 2
    ) -> List[Tuple[str, float]]:
        """Sample k programs from island: cluster (signature) then program (short bias)."""
        samples = []
        for _ in range(k):
            # Cluster by signature
            clusters: Dict[tuple, List[Tuple[str, float]]] = defaultdict(list)
            for prog, score, sig in island:
                clusters[sig].append((prog, score))
            if not clusters:
                continue
            cluster_sigs = list(clusters.keys())
            cluster_means = [np.mean(sig) for sig in cluster_sigs]
            # Boltzmann probs on cluster means
            exp_terms = np.exp(np.array(cluster_means) / self.T_cluster)
            probs = exp_terms / exp_terms.sum()
            # Sample cluster index (FIX: always 1D numeric indices)
            indices = list(range(len(cluster_sigs)))
            chosen_idx = np.random.choice(indices, p=probs)
            chosen_sig = cluster_sigs[chosen_idx]
            cluster_progs = clusters[chosen_sig]
            # Sample program in cluster: bias shorter (negative length)
            lengths = [-len(prog) for prog, _ in cluster_progs]
            min_l = min(lengths)
            max_l = max(lengths)
            if max_l == min_l:
                chosen_prog, chosen_score = random.choice(cluster_progs)
            else:
                norm_l = [(l - min_l) / (max_l - min_l + 1e-6) for l in lengths]  # noqa: E741
                exp_terms_p = np.exp(np.array(norm_l) / self.T_program)
                probs_p = exp_terms_p / exp_terms_p.sum()
                idx = np.random.choice(len(cluster_progs), p=probs_p)
                chosen_prog, chosen_score = cluster_progs[idx]
            samples.append((chosen_prog, chosen_score))
        return samples

    def sample_k_programs(self, k: int = 2) -> Tuple[int, List[Tuple[str, float]]]:
        """Sample island then k programs from it."""
        non_empty_islands = [i for i, isl in enumerate(self.islands) if len(isl) > 0]
        if not non_empty_islands:
            raise ValueError("No programs in any island")
        island_idx = np.random.choice(non_empty_islands)
        island = self.islands[island_idx]
        samples = self.sample_k_from_island(island, k)
        return island_idx, samples

    def add_program(self, island_idx: int, prog_code: str, score: float, sig: tuple):
        """Add valid program to island (prune if full)."""
        island = self.islands[island_idx]
        if len(island) >= self.max_island_size:
            # Prune lowest score
            min_idx = min(range(len(island)), key=lambda i: island[i][1])
            island.pop(min_idx)
        island.append((prog_code, score, sig))
        # Update global best
        if score > (self.best_overall[1] if self.best_overall else -np.inf):
            self.best_overall = (prog_code, score, sig)
            print(f"New best mean: {score:.2f}, sizes: {[int(round(s)) for s in sig]}")

    def reset_islands(self):
        """Island evolution: kill worst half, seed from best survivors."""
        island_bests = [
            (max((p[1] for p in island), default=-np.inf), i)
            for i, island in enumerate(self.islands)
        ]
        island_bests.sort(reverse=True)
        num_survive = self.num_islands // 2
        survivors = [ib[1] for ib in island_bests[:num_survive]]
        for i in range(num_survive, self.num_islands):
            self.islands[i] = []
            if survivors:
                surv_island = random.choice(survivors)
                best_entry = max(self.islands[surv_island], key=lambda x: x[1])
                self.islands[i].append(best_entry)
        print("Islands reset: worst half reseeded from elites.")


# LLM sampler (mock for testing; uncomment OpenAI for real)
def mock_llm_sample(prompt: str, temperature: float = 0.7) -> str:
    """Mock LLM: simple mutations/heurs for testing (no API needed)."""
    # Extract a base from v0/v1 if possible
    match = re.search(
        r'def priority_v[0-9]\([^)]*\):\s*""".*?"""\s*(.*?return\s+.*?)(?=\n\n|\Z)',
        prompt,
        re.DOTALL | re.MULTILINE,
    )
    base = match.group(1).strip() if match else "return 0.0"
    mutations = [
        "return sum(element)",
        "return sum(x * (i+1) for i, x in enumerate(element))",
        "score = 0.0\nfor i in range(n):\n    score += element[i] * (n - i)\nreturn score",
        "return n - element.count(0)",
        "score = element.count(1) * 2 + element.count(2)\nreturn score",
        "if element[0] == 0: return 10.0\nreturn 0.0",
        base.replace("0.0", "1.0"),  # Trivial tweak
        "return len(element) - sum(element)",  # More zeros lower prio?
    ]
    body = random.choice(mutations)
    return f"    {body}\n"  # Indent for exec


# Real LLM (uncomment, set OPENAI_API_KEY)
# def llm_sample(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
#     import openai
#     client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Expert Python coder. COMPLETE ONLY the indented BODY of priority_v2 (after docstring). Output valid, indented Python. NO new defs/imports/signatures. Focus on clever priority for large cap sets."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens
#     )
#     return response.choices[0].message.content.strip()


# Main FunSearch loop (paper's algorithm)
def run_funsearch(
    num_iterations: int = 1000, samples_per_prompt: int = 10, use_mock: bool = True
):
    """Run FunSearch for cap set problem."""
    db = ProgramsDB(num_islands=20)
    llm_func = mock_llm_sample  # if use_mock else llm_sample
    print("Starting FunSearch. Best known:", KNOWN_BESTS)
    print("Press Ctrl+C to stop.\n")

    for it in range(num_iterations):
        try:
            island_idx, prog_samples = db.sample_k_programs(k=2)
            # Sort ascending score: v0 (worse), v1 (better) per paper
            prog_samples.sort(key=lambda x: x[1])
            v_codes = []
            for vi, (prog_code, _) in enumerate(prog_samples):
                v_source = prog_code.replace("def priority(", f"def priority_v{vi}(")
                v_codes.append(v_source)
            # Best-shot prompt: prefix + v0 + v1 + empty v2 header
            v2_header = f"\ndef priority_v2(element: tuple[int, ...], n: int) -> float:\n    {PRIORITY_DOC}\n    "
            prompt = PROMPT_PREFIX + "\n\n".join(v_codes) + v2_header
            # Generate & evaluate multiple samples per prompt
            for _ in range(samples_per_prompt):
                response = llm_func(prompt)
                full_generated = prompt + response
                try:
                    env_parse = EVAL_ENV.copy()
                    exec(full_generated, env_parse)
                    if "priority_v2" not in env_parse:
                        continue
                    v2_func = env_parse["priority_v2"]
                    v2_source = inspect.getsource(v2_func).strip()
                    # Extract & rename to 'priority'
                    new_prog_code = re.sub(r"priority_v2", "priority", v2_source)
                    # Clean pass/empty
                    new_prog_code = re.sub(
                        r"^\s*pass\s*\n?", "", new_prog_code, flags=re.MULTILINE
                    )
                    score, sig = db._evaluate_program(new_prog_code)
                    if score is not None:
                        db.add_program(island_idx, new_prog_code, score, sig)
                        db.total_evaluations += 1
                except Exception:
                    continue
            # Periodic island reset
            if db.total_evaluations % db.reset_every == 0 and db.total_evaluations > 0:
                db.reset_islands()
            # Progress
            if (it + 1) % 100 == 0:
                if db.best_overall:
                    mean, sig = db.best_overall[1], db.best_overall[2]
                    sizes = [int(round(s)) for s in sig]
                    print(
                        f"Iter {it+1}, evals {db.total_evaluations}, best mean: {mean:.2f}, sizes: {sizes}"
                    )
                    # Check improvements
                    improved_dims = [
                        INPUTS[i]
                        for i in range(len(sig))
                        if sizes[i] > KNOWN_BESTS[INPUTS[i]]
                    ]
                    if improved_dims:
                        print(f"*** NEW RECORDS in dims {improved_dims}! ***")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
    # Final results
    if db.best_overall:
        print("\n=== FINAL BEST PROGRAM ===")
        print(db.best_overall[0])
        print(f"Mean size: {db.best_overall[1]:.2f}")
        sizes = [int(round(s)) for s in db.best_overall[2]]
        print(f"Sizes: {dict(zip(INPUTS, sizes))}")
        print("========================")
    return db.best_overall


if __name__ == "__main__":
    # Test run (mock LLM, fast)
    best = run_funsearch(num_iterations=1000, samples_per_prompt=5, use_mock=True)
    # For real LLM: uncomment llm_sample imports/setup, use_mock=False, ~10k+ iters to approach paper results (stochastic)
