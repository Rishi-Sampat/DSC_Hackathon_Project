import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import load_dataset, preprocess_dataset, get_training_sets

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# -------------------------------------------------
# BUILD TF-IDF FEATURES
# -------------------------------------------------
def build_tfidf_features(X_text):
    """
    Builds and fits a TF-IDF vectorizer on input text.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # unigrams + bigrams
        min_df=2,                  # ignore very rare terms
        max_df=0.9,                # ignore overly common terms
        stop_words="english",
        sublinear_tf=True
    )

    X_tfidf = vectorizer.fit_transform(X_text)

    return vectorizer, X_tfidf


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    # Load and preprocess dataset
    df = load_dataset()
    df_clean = preprocess_dataset(df)

    # Get text features
    (
        X_text,
        _,
        _,
        _,
        _
    ) = get_training_sets(df_clean)

    # Build TF-IDF
    vectorizer, X_tfidf = build_tfidf_features(X_text)

    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("TF-IDF feature matrix shape:", X_tfidf.shape)
    print("TF-IDF vectorizer saved to:", VECTORIZER_PATH)

    # Show some feature names (sanity check)
    feature_names = vectorizer.get_feature_names_out()
    print("\nSample features:")
    print(feature_names[:20])
