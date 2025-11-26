import random
import re


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
