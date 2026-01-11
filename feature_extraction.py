from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_features(
    texts,
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2
):
    """
    Build TF-IDF features optimized for short factual statements.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )

    X_tfidf = vectorizer.fit_transform(texts)
    return vectorizer, X_tfidf
