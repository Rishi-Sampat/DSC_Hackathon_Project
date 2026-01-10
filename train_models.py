import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_preprocessing import load_dataset, preprocess_dataset, get_training_sets
from feature_extraction import build_tfidf_features

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_DIR = "./"

# -------------------------------------------------
# LOAD & PREPARE DATA
# -------------------------------------------------
df = load_dataset()
df_clean = preprocess_dataset(df)

(
    X_text,
    y_h_flag,
    y_h_type,
    y_b_flag,
    y_b_type
) = get_training_sets(df_clean)

# -------------------------------------------------
# TF-IDF FEATURES
# -------------------------------------------------
vectorizer, X_tfidf = build_tfidf_features(X_text)

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_h_flag_train, y_h_flag_test = train_test_split(
    X_tfidf, y_h_flag, test_size=0.2, random_state=42, stratify=y_h_flag
)

_, _, y_h_type_train, y_h_type_test = train_test_split(
    X_tfidf, y_h_type, test_size=0.2, random_state=42, stratify=y_h_flag
)

_, _, y_b_flag_train, y_b_flag_test = train_test_split(
    X_tfidf, y_b_flag, test_size=0.2, random_state=42, stratify=y_b_flag
)

_, _, y_b_type_train, y_b_type_test = train_test_split(
    X_tfidf, y_b_type, test_size=0.2, random_state=42, stratify=y_b_flag
)

# -------------------------------------------------
# MODEL DEFINITIONS
# -------------------------------------------------
def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print("F1-score:", f1_score(y_te, y_pred, average="weighted"))
    print(classification_report(y_te, y_pred))

    return model

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
hallucination_flag_model = train_and_evaluate(
    LogisticRegression(max_iter=1000),
    X_train, X_test,
    y_h_flag_train, y_h_flag_test,
    "Hallucination Flag Model"
)

hallucination_type_model = train_and_evaluate(
    LogisticRegression(max_iter=1000),
    X_train, X_test,
    y_h_type_train, y_h_type_test,
    "Hallucination Type Model"
)

bias_flag_model = train_and_evaluate(
    LogisticRegression(max_iter=1000),
    X_train, X_test,
    y_b_flag_train, y_b_flag_test,
    "Bias Flag Model"
)

bias_type_model = train_and_evaluate(
    LogisticRegression(max_iter=1000),
    X_train, X_test,
    y_b_type_train, y_b_type_test,
    "Bias Type Model"
)

# -------------------------------------------------
# SAVE MODELS
# -------------------------------------------------
joblib.dump(vectorizer, MODEL_DIR + "tfidf_vectorizer.pkl")
joblib.dump(hallucination_flag_model, MODEL_DIR + "hallucination_flag_model.pkl")
joblib.dump(hallucination_type_model, MODEL_DIR + "hallucination_type_model.pkl")
joblib.dump(bias_flag_model, MODEL_DIR + "bias_flag_model.pkl")
joblib.dump(bias_type_model, MODEL_DIR + "bias_type_model.pkl")

print("\nAll models trained and saved successfully.")
