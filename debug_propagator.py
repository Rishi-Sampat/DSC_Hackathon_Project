from multi_claim_splitter import split_claims
from claim_propagator import propagate_subject

claims = split_claims(
    "Einstein was German and died in Princeton"
)

print(propagate_subject(claims))