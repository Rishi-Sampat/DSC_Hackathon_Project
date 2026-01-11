import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_preprocessing import load_dataset, preprocess_dataset, get_training_sets
from feature_extraction import build_tfidf_features

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
# TF-IDF FEATURES (OPTIMIZED FOR SHORT SENTENCES)
# -------------------------------------------------
vectorizer, X_tfidf = build_tfidf_features(
    X_text,
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2
)

# -------------------------------------------------
# SINGLE CONSISTENT SPLIT
# -------------------------------------------------
indices = np.arange(X_tfidf.shape[0])

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y_h_flag
)

X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
y_h_flag_train, y_h_flag_test = y_h_flag[train_idx], y_h_flag[test_idx]
y_b_flag_train, y_b_flag_test = y_b_flag[train_idx], y_b_flag[test_idx]

# -------------------------------------------------
# HALLUCINATION TYPE (ONLY WHERE hallucination == 1)
# -------------------------------------------------
mask_train_h = y_h_flag_train == 1
mask_test_h = y_h_flag_test == 1

X_train_h = X_train[mask_train_h]
X_test_h = X_test[mask_test_h]

y_h_type_train = y_h_type[train_idx][mask_train_h]
y_h_type_test = y_h_type[test_idx][mask_test_h]

# -------------------------------------------------
# BIAS TYPE (ONLY WHERE bias == 1)
# -------------------------------------------------
mask_train_b = y_b_flag_train == 1
mask_test_b = y_b_flag_test == 1

X_train_b = X_train[mask_train_b]
X_test_b = X_test[mask_test_b]

y_b_type_train = y_b_type[train_idx][mask_train_b]
y_b_type_test = y_b_type[test_idx][mask_test_b]

# -------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------
def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, name):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_te, preds))
    print("F1-score:", f1_score(y_te, preds, average="weighted"))
    print(classification_report(y_te, preds))

    return model

# -------------------------------------------------
# MODELS (BALANCED)
# -------------------------------------------------
hallucination_flag_model = train_and_evaluate(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    X_train, X_test,
    y_h_flag_train, y_h_flag_test,
    "Hallucination Flag Model"
)

hallucination_type_model = train_and_evaluate(
    LogisticRegression(max_iter=2000),
    X_train_h, X_test_h,
    y_h_type_train, y_h_type_test,
    "Hallucination Type Model"
)

bias_flag_model = train_and_evaluate(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    X_train, X_test,
    y_b_flag_train, y_b_flag_test,
    "Bias Flag Model"
)

bias_type_model = train_and_evaluate(
    LogisticRegression(max_iter=2000),
    X_train_b, X_test_b,
    y_b_type_train, y_b_type_test,
    "Bias Type Model"
)

# -------------------------------------------------
# SAVE ARTIFACTS
# -------------------------------------------------
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(hallucination_flag_model, "hallucination_flag_model.pkl")
joblib.dump(hallucination_type_model, "hallucination_type_model.pkl")
joblib.dump(bias_flag_model, "bias_flag_model.pkl")
joblib.dump(bias_type_model, "bias_type_model.pkl")

print("\nAll models retrained and saved successfully.")
