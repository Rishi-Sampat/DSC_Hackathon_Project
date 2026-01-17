import subprocess
import json


def ollama_judge(statement: str) -> dict:
    """
    Uses local Ollama LLM for commonsense reasoning.
    Returns a structured judgment compatible with pipeline.py
    """

    prompt = f"""
You are an expert fact checker and bias analyst.

Judge the following statement using common sense and general world knowledge.

Rules:
- Ignore rare edge cases unless explicitly stated.
- Be conservative and realistic.
- If generally false, verdict = false
- If generally true, verdict = true
- If partially true or context-dependent, verdict = misleading
- If cannot be judged, verdict = unverifiable

Bias rules:
- If the statement stereotypes or unfairly generalizes a group, bias = yes
- Otherwise, bias = no
- If bias = yes, choose ONE bias_type from:
  gender, social, racial, ethical, political

Respond ONLY in valid JSON with EXACTLY these fields:
- verdict
- reasoning
- corrected_statement
- bias
- bias_type

Statement:
"{statement}"
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )

        raw_output = result.stdout.strip()

        # Safely extract JSON block
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON found in Ollama output")

        json_text = raw_output[start:end]
        data = json.loads(json_text)

        # ---- HARD SAFETY NORMALIZATION ----
        verdict = data.get("verdict", "unverifiable")
        bias = data.get("bias", "no")
        bias_type = data.get("bias_type", "none")

        if bias != "yes":
            bias_type = "none"

        return {
            "verdict": verdict,
            "reasoning": data.get("reasoning", ""),
            "corrected_statement": data.get("corrected_statement", statement),
            "bias": bias,
            "bias_type": bias_type
        }

    except Exception:
        # Absolute safety fallback (pipeline-safe)
        return {
            "verdict": "unverifiable",
            "reasoning": "Local LLM unavailable or response malformed.",
            "corrected_statement": statement,
            "bias": "no",
            "bias_type": "none"
        }
