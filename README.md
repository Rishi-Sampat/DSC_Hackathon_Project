Multi-Stage Fact Verification & Bias Detection System | DSC Hackathon | Jan 2026
- Designed and implemented hierarchical verification pipeline integrating four knowledge sources:
  (1) Local fact databases (temporal, numeric, causal, comparative facts),
  (2) Wikidata entity linking and resolution,
  (3) Wikipedia evidence retrieval with summarization,
  (4) Ollama LLM commonsense reasoning for unverifiable/partially true claims
- Engineered NLP components:
  • Claim normalizer: Regex + semantic analysis for 15+ relation types 
    (capital_of, born_in, occupied_in, causes, etc.)
  • ML feature extraction: TF-IDF vectorization + scikit-learn classifiers for 
    hallucination/bias binary and multi-class detection
  • Multi-claim orchestration: AND/OR aggregation logic with subject propagation 
    for compound statements
- Implemented conflict resolution hierarchy: fact DB > external APIs > LLM > ML models
- Created test harness: ~20 debug scripts for component-level testing; results 
  exported to Excel for manual validation
- Stack: Python 3, scikit-learn, joblib, Wikidata/Wikipedia APIs, Ollama, regex
- Demonstrated end-to-end reasoning: Accepts natural language input → outputs 
  structured dict with truth status, hallucination/bias flags, sources, and 
  corrected statement
