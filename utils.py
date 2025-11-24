from sampler import Sampler
from evaluation import Evaluator
from database import ProgramsDatabase, Program

def funsearch_worker(sampler: Sampler, evaluator: Evaluator, database: ProgramsDatabase, island_id: int):
    """
    Represents a single thread of execution:
    Prompt -> LLM -> Evaluate -> Database
    """
    # 1. Generate new code
    # We pass the function name we want to evolve (e.g., 'priority')
    new_code_snippets = sampler.sample(database, "priority")

    for snippet in new_code_snippets:
        # Wrap the snippet in a full function definition for execution
        # (The Sampler prompt ended at the header, LLM gave the body)
        full_program = f"def priority_vX(element, n):\n{snippet}"

        # 2. Evaluate
        score, signature = evaluator.evaluate_program(full_program)

        # 3. Register (if valid)
        if score > float('-inf'):
            prog = Program(code=full_program, score=score, signature=signature)
            database.register_program(prog, island_id)
            print(f"Island {island_id}: Registered program with score {score}")