from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random
import time
from loguru import logger


@dataclass
class Program:
    code: str
    score: float
    signature: Tuple


@dataclass
class Island:
    """
    An isolated subpopulation of programs.
    FunSearch periodically resets the worst half of islands[cite: 130].
    """

    clusters: Dict[Tuple, List[Program]] = field(default_factory=dict)

    def add_program(self, program: Program):
        if program.signature not in self.clusters:
            self.clusters[program.signature] = []
        self.clusters[program.signature].append(program)

    def get_best_score(self) -> float:
        """Returns the score of the best individual in this island."""
        if not self.clusters:
            return -float("inf")
        return max(p.score for cluster in self.clusters.values() for p in cluster)


class ProgramsDatabase:
    def __init__(self, num_islands=10):
        self.islands = [Island() for _ in range(num_islands)]
        self.reset_interval = 4 * 60 * 60  # Paper uses 4 hours [cite: 636]
        self.last_reset = time.time()

    def register_program(self, program: Program, island_id: int):
        """Stores a successfully evaluated program in its specific island."""
        self.islands[island_id].add_program(program)

    def get_prompt_programs(self, island_id: int, k=2) -> List[str]:
        """
        Selects k programs from an island to put in the prompt.
        Selection logic:
        1. Sample a cluster proportional to score (Boltzmann selection).
        2. Sample a program within cluster favoring shorter code.
        [cite: 649, 657]
        """
        island = self.islands[island_id]
        if not island.clusters:
            return []  # Should handle initialization logic

        selected_programs = []
        clusters = list(island.clusters.values())

        # Simplified selection logic for demo (replace with Boltzmann in production)
        # Sort clusters by max score of their contents
        clusters.sort(key=lambda c: max(p.score for p in c), reverse=True)

        # Pick top k clusters, then the best program from each
        for i in range(min(k, len(clusters))):
            cluster = clusters[i]
            # Paper favors shorter programs within clusters [cite: 657]
            best_prog = sorted(cluster, key=lambda p: (-p.score, len(p.code)))[0]
            selected_programs.append(best_prog.code)

        return selected_programs

    def reset_islands(self):
        """
        Periodically discards worst half of islands and reseeds them
        from the best half[cite: 130, 131].
        """
        # Sort islands by their best program's score
        sorted_indices = sorted(
            range(len(self.islands)), key=lambda i: self.islands[i].get_best_score()
        )

        num_to_reset = len(self.islands) // 2
        worst_indices = sorted_indices[:num_to_reset]
        best_indices = sorted_indices[num_to_reset:]

        logger.info(f"Resetting islands: {worst_indices} seeded from {best_indices}")

        for bad_idx in worst_indices:
            # Clone a random "best" island
            _ = random.choice(best_indices)
            # In a real impl, deep copy the data. Here we just re-init.
            self.islands[bad_idx] = Island()
            # (Implementation Detail: You would copy the best program to the new island)
