from models import ModelManager
from factory import ProviderFactory

mm = ModelManager()


def _extract_cpp(text: str) -> str:
    """
    Extract raw C++ from an LLM response.
    Handles fenced Markdown blocks and strips common non-code trailers.
    """
    if not text:
        return ""

    s = text.strip()

    # Prefer the first fenced block if present
    if "```" in s:
        parts = s.split("```")
        # parts[1] is the first fenced block content, possibly starting with "cpp\n"
        if len(parts) >= 3:
            block = parts[1].lstrip()
            # Drop an optional language tag line (e.g. "cpp")
            first_nl = block.find("\n")
            if first_nl != -1:
                first_line = block[:first_nl].strip().lower()
                if first_line in {"cpp", "c++", "cc", "c"}:
                    block = block[first_nl + 1 :]
            return block.strip()

    # No fences: heuristically stop when prose begins after code.
    lines = s.splitlines()
    cleaned = []
    for line in lines:
        if line.strip().startswith("In this implementation"):
            break
        if line.strip().startswith("Explanation"):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def generate_cpp(provider: str, model: str, expertise: str, python_code_or_prompt: str) -> str:
    client = ProviderFactory.create(provider)
    system = mm.system_prompt(expertise)
    prompt = (
        "Convert the following into clean, modern C++.\n"
        "- Output ONLY valid C++ code.\n"
        "- Do NOT use Markdown fences (no ```).\n"
        "- Do NOT include explanations, comments outside code, or any extra text.\n"
        "- Prefer C++17.\n"
        "- If input is a description, implement it.\n\n"
        f"INPUT:\n{python_code_or_prompt}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or ""
    return _extract_cpp(raw)
