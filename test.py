from claim_normalizer import normalize_claim
from verifier_semantic import verify_structured_claim

tests = [
    "Delhi is the capital of India",
    "India has 8 union territories",
    "Rajkot is the capital of India"
]

for t in tests:
    claim = normalize_claim(t)
    print("\nInput:", t)
    print("Claim:", claim)

    if claim["type"] == "structured":
        status, sources = verify_structured_claim(claim)
        print("Truth Status:", status)
        print("Sources:", sources)
