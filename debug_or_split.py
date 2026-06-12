# debug_or_split.py

from multi_claim_splitter import split_claims

claims, connectors = split_claims(
    "Hitler was Austrian and died in Berlin"
)

print(claims)
print(connectors)