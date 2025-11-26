# import openai  # Uncomment and pip install openai for real LLM
from search import run_funsearch

if __name__ == "__main__":
    # Test run (mock LLM, fast)
    best = run_funsearch(num_iterations=1000, samples_per_prompt=5, use_mock=True)
    # For real LLM: uncomment llm_sample imports/setup, use_mock=False, ~10k+ iters to approach paper results (stochastic)
