from database import ProgramsDB
from sampler import mock_llm_sample
from constants import KNOWN_BESTS, PRIORITY_DOC, PROMPT_PREFIX, EVAL_ENV, INPUTS
import re
import inspect
from loguru import logger


# Main FunSearch loop (paper's algorithm)
def run_funsearch(
    num_iterations: int = 1000, samples_per_prompt: int = 10, use_mock: bool = True
):
    """Run FunSearch for cap set problem."""
    db = ProgramsDB(num_islands=20)
    llm_func = mock_llm_sample  # if use_mock else llm_sample
    logger.info("Starting FunSearch. Best known:", KNOWN_BESTS)
    logger.info("Press Ctrl+C to stop.\n")

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
                    logger.info(
                        f"Iter {it+1}, evals {db.total_evaluations}, best mean: {mean:.2f}, sizes: {sizes}"
                    )
                    # Check improvements
                    improved_dims = [
                        INPUTS[i]
                        for i in range(len(sig))
                        if sizes[i] > KNOWN_BESTS[INPUTS[i]]
                    ]
                    if improved_dims:
                        logger.info(f"*** NEW RECORDS in dims {improved_dims}! ***")
        except KeyboardInterrupt:
            logger.info("\nStopped by user.")
            break
    # Final results
    if db.best_overall:
        logger.info("\n=== FINAL BEST PROGRAM ===")
        logger.info(db.best_overall[0])
        logger.info(f"Mean size: {db.best_overall[1]:.2f}")
        sizes = [int(round(s)) for s in db.best_overall[2]]
        logger.info(f"Sizes: {dict(zip(INPUTS, sizes))}")
        logger.info("========================")
    return db.best_overall
