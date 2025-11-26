from collections import defaultdict
import numpy as np
import random
from loguru import logger
from constants import EVAL_ENV, INITIAL_PRIORITY_CODE, INPUTS
from utils import solve
from eval import evaluate_capset


class ProgramsDB:
    def __init__(self, num_islands=20, max_island_size=2000):
        self.num_islands = num_islands
        self.islands = [[] for _ in range(num_islands)]
        self.max_island_size = max_island_size
        self.total_evaluations = 0
        self.best_overall = None
        self.inputs = INPUTS
        self.T_cluster = 0.25  # Tuned empirically; lower favors best more
        self.T_program = 0.05  # Favor shorter programs strongly
        self.reset_every = 512  # Evaluations per reset (simulate 4h with scaling)
        # Initialize islands with initial program
        init_score, init_sig = self._evaluate_program(INITIAL_PRIORITY_CODE)
        init_entry = (INITIAL_PRIORITY_CODE, init_score, init_sig)
        for island in self.islands:
            island.append(init_entry)
        self.best_overall = init_entry
        logger.info(f"Initialized with initial score: {str(init_score)} (mean size)")

    def _evaluate_program(self, prog_code):
        """Evaluate program on all inputs, return mean score and signature."""
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
                if size is None:
                    return None, None
                scores.append(float(size))
            mean_score = np.mean(scores)
            signature = tuple(scores)
            return mean_score, signature
        except Exception:
            return None, None

    def sample_k_from_island(self, island, k=2):
        """Sample k programs from a single island using cluster + length biasing."""
        samples = []
        for _ in range(k):
            # Build clusters by signature
            clusters = defaultdict(list)
            for prog, score, sig in island:
                clusters[sig].append((prog, score))
            if not clusters:
                continue
            cluster_sigs = list(clusters.keys())
            cluster_means = [np.mean(sig) for sig in cluster_sigs]
            # Boltzmann on cluster scores
            exp_terms = np.exp(np.array(cluster_means) / self.T_cluster)
            probs = exp_terms / exp_terms.sum()
            chosen_sig = np.random.choice(cluster_sigs, p=probs)
            cluster_progs = clusters[chosen_sig]
            # Sample program favoring shorter
            lengths = [-len(prog) for prog, _ in cluster_progs]
            min_l, max_l = min(lengths), max(lengths)
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

    def sample_k_programs(self, k=2):
        """Sample island and k programs from it."""
        # Sample non-empty island
        non_empty_islands = [i for i, isl in enumerate(self.islands) if len(isl) > 0]
        if not non_empty_islands:
            raise ValueError("No programs in DB")
        island_idx = np.random.choice(non_empty_islands)
        island = self.islands[island_idx]
        samples = self.sample_k_from_island(island, k)
        return island_idx, samples

    def add_program(self, island_idx, prog_code, score, sig):
        """Add program to island if valid."""
        island = self.islands[island_idx]
        # Prune if too large: keep best
        if len(island) >= self.max_island_size:
            # Remove lowest score
            island.pop(min(range(len(island)), key=lambda i: island[i][1]))
        island.append((prog_code, score, sig))
        # Update best overall
        if score > (self.best_overall[1] if self.best_overall else -np.inf):
            self.best_overall = (prog_code, score, sig)
            logger.info(f"New best mean score: {score:.2f}, signature: {sig}")

    def reset_islands(self):
        """Reset worst half islands."""
        island_bests = []
        for i, island in enumerate(self.islands):
            if island:
                best_score = max(p[1] for p in island)
                island_bests.append((best_score, i))
            else:
                island_bests.append((-np.inf, i))
        island_bests.sort(reverse=True)
        num_survive = self.num_islands // 2
        survivors = [ib[1] for ib in island_bests[:num_survive]]
        for i in range(num_survive, self.num_islands):
            self.islands[i] = []
            if survivors:
                surv_island = random.choice(survivors)
                best_entry = max(self.islands[surv_island], key=lambda x: x[1])
                self.islands[i].append(best_entry)
        logger.info("Islands reset.")
