import os
import joblib

from claim_normalizer import normalize_claim
from verifier_semantic import verify_structured_claim
from evidence_wikipedia import query_wikipedia_summary
from statement_classifier import classify_statement
from contradiction_checker import check_contradiction
from ollama_reasoner import ollama_judge


# -------------------------------
# LOAD MODELS (SAFE PATHS)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
hallucination_flag_model = joblib.load(os.path.join(BASE_DIR, "hallucination_flag_model.pkl"))
hallucination_type_model = joblib.load(os.path.join(BASE_DIR, "hallucination_type_model.pkl"))
bias_flag_model = joblib.load(os.path.join(BASE_DIR, "bias_flag_model.pkl"))
bias_type_model = joblib.load(os.path.join(BASE_DIR, "bias_type_model.pkl"))


def run_pipeline(input_text: str) -> dict:
    output = {
        "input_statement": input_text,
        "hallucination_detected": False,
        "hallucination_type": "none",
        "bias_detected": False,
        "bias_type": "none",
        "truth_status": "Unverifiable",
        "corrected_statement": "",
        "sources": [],
        "explanation": ""
    }

    # -------------------------------
    # 1. NORMALIZATION
    # -------------------------------
    claim = normalize_claim(input_text)
    statement_type = classify_statement(input_text)

    # -------------------------------
    # 2. ML (RISK ESTIMATION ONLY)
    # -------------------------------
    X = tfidf.transform([input_text])

    hallucination_pred = hallucination_flag_model.predict(X)[0]
    hallucination_type_pred = hallucination_type_model.predict(X)[0]

    bias_pred = bias_flag_model.predict(X)[0]
    bias_type_pred = bias_type_model.predict(X)[0]

    output["bias_detected"] = bool(bias_pred)
    output["bias_type"] = bias_type_pred if bias_pred else "none"

    # -------------------------------
    # 3. FACT VERIFICATION
    # -------------------------------
    sources = []

    if claim["type"] == "structured":
        truth_status, sources = verify_structured_claim(claim)
        if truth_status != "False" and statement_type == "HARD_FACT":
            truth_status = "True"
    else:
        wiki = query_wikipedia_summary(input_text)
        if wiki:
            truth_status = "Partially true"
            sources = [wiki]
        else:
            truth_status = "Unverifiable"

    output["truth_status"] = truth_status
    output["sources"] = sources

    # -------------------------------
    # 4. CONTRADICTION CHECK
    # -------------------------------
    if sources:
        if check_contradiction(input_text, sources[0]["text"]):
            output["truth_status"] = "False"
            output["hallucination_detected"] = True
            output["hallucination_type"] = "factual"

    # -------------------------------
    # 5. OLLAMA COMMONSENSE OVERRIDE
    # -------------------------------
    if output["truth_status"] in ["Unverifiable", "Partially true"] or statement_type in ["COMPARATIVE", "UNVERIFIABLE"]:
        llm = ollama_judge(input_text)

        if llm["verdict"] == "false":
            output["truth_status"] = "False"
            output["hallucination_detected"] = True
            output["hallucination_type"] = "factual"
            output["corrected_statement"] = llm["corrected_statement"]

        elif llm["verdict"] == "true":
            output["truth_status"] = "True"
            output["hallucination_detected"] = False
            output["hallucination_type"] = "none"

        elif llm["verdict"] == "misleading":
            output["truth_status"] = "Misleading"
            output["hallucination_detected"] = True
            output["hallucination_type"] = "logical"
            output["corrected_statement"] = llm["corrected_statement"]

        if llm["bias"] == "yes":
            output["bias_detected"] = True

        output["explanation"] += " Commonsense reasoning applied via local LLM."

    # -------------------------------
    # 6. FINAL CORRECTION
    # -------------------------------
    if not output["corrected_statement"]:
        if output["bias_detected"]:
            output["corrected_statement"] = "This statement contains bias and should be rephrased neutrally."
        elif output["hallucination_detected"]:
            output["corrected_statement"] = (
                sources[0]["text"] if sources else "This claim is generally false or unverifiable."
            )
        else:
            output["corrected_statement"] = "No correction required."

    # -------------------------------
    # 7. EXPLANATION
    # -------------------------------
    output["explanation"] = (
        f"Statement type: {statement_type}. "
        f"Truth status: {output['truth_status']}. "
        f"Bias detected: {output['bias_detected']}. "
        f"Hallucination detected: {output['hallucination_detected']}. "
        f"Decision made using ML risk estimation, fact verification, and local LLM commonsense reasoning."
    )

    return output
