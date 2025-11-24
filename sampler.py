from llm import LLM
from typing import List


def build_prompt(previous_programs: List[str], function_to_evolve: str) -> str:
    """
    Constructs 'Best-Shot Prompting' as described in the paper.
    Concatenates k programs to help LLM see patterns[cite: 181].
    """
    prompt = "--- PAGE 1 ---\n"  # Context header
    version_count = 0

    # Append existing high-scoring programs (v0, v1...)
    for code in previous_programs:
        # Rename the function to versioned names (e.g., priority_v0)
        code_versioned = code.replace(f"def {function_to_evolve}(", f"def {function_to_evolve}_v{version_count}(")
        prompt += code_versioned + "\n\n"
        version_count += 1

    # Append the header for the new function the LLM needs to write
    prompt += f"# Improved version of {function_to_evolve}_v{version_count-1}\n"
    prompt += f"def {function_to_evolve}_v{version_count}(element, n):\n"

    return prompt


class Sampler:
    """
    Generates prompts and queries the LLM.
    """
    def __init__(self, llm: LLM):
        self.llm = llm

    def sample(self, database, function_name: str):
        """
        Main inference loop:
        1. Get parents from DB
        2. Build Prompt
        3. Query LLM
        """
        # 1. Retrieve k parents (usually k=2) from a single island [cite: 134]
        parents = database.get_prompt_programs()

        # 2. Build prompt
        prompt = build_prompt(parents, function_name)

        # 3. Get samples (LLM completes the code)
        samples = self.llm.draw_samples(prompt)

        return samples