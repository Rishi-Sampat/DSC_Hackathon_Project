# debug_or_propagator.py

from claim_propagator import propagate_subject

claims = [
    "Hitler was Austrian",
    "German"
]

print(
    propagate_subject(claims)
)