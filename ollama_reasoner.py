import subprocess
import json


def ollama_judge(statement: str) -> dict:
    """
    Uses local Ollama LLM for commonsense reasoning.
    Returns a structured judgment.
    """

    prompt = f"""
You are an expert fact checker.

Judge the following statement using common sense and general world knowledge.

Rules:
- Ignore rare edge cases unless explicitly mentioned.
- Be conservative.
- If generally false, say false.
- If generally true, say true.
- If depends on context, say misleading.
- If cannot be judged, say unverifiable.

Respond ONLY in valid JSON with fields:
- verdict: true / false / misleading / unverifiable
- reasoning: short explanation
- corrected_statement: corrected or clarified version
- bias: yes or no

Statement:
"{statement}"
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()

        # Extract JSON safely
        start = output.find("{")
        end = output.rfind("}") + 1
        json_text = output[start:end]

        return json.loads(json_text)

    except Exception:
        # Absolute safety fallback
        return {
            "verdict": "unverifiable",
            "reasoning": "Local LLM unavailable.",
            "corrected_statement": statement,
            "bias": "no"
        }
