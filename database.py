from collections import defaultdict
import numpy as np
import random
from loguru import logger
from constants import EVAL_ENV, INITIAL_PRIORITY_CODE, INPUTS
from utils import solve
from eval import evaluate_capset
from typing import List, Tuple, Dict, Optional


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
        if init_score is not None:
            logger.info(
                f"Initialized with initial score: {init_score:.2f} (mean size), signature: {[int(s) for s in init_sig]}"
            )
        else:
            logger.warning(
                "Failed to evaluate initial program. Starting with None score."
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
                # Only include programs with valid signatures
                if sig is not None and all(s is not None for s in sig):
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
            logger.info(
                f"New best mean: {score:.2f}, sizes: {[int(round(s)) for s in sig]}"
            )

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
        logger.info("Islands reset: worst half reseeded from elites.")
