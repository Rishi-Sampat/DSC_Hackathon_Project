import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_PATH = DATASET_PATH = r"E:\College\DSC_Hack_hybrid\ffff_final.xlsx"   # adjust if needed

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
def load_dataset(path=DATASET_PATH):
    df = pd.read_excel(path)

    # Rename columns (safety)
    df.columns = df.columns.str.strip().str.lower()

    return df


# -------------------------------------------------
# SELECT & CLEAN REQUIRED COLUMNS
# -------------------------------------------------
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only dataset-aligned columns and removes noisy / unused ones.
    """

    required_columns = [
        "ai_response",
        "topic",
        "label_hallucination",
        "hallucination_type",
        "label_bias",
        "bias_type",
        "corrected_response"
    ]

    # Keep only required columns
    df = df[required_columns]

    # Drop rows with missing labels
    df = df.dropna(
        subset=[
            "ai_response",
            "label_hallucination",
            "label_bias"
        ]
    )

    # Ensure correct data types
    df["label_hallucination"] = df["label_hallucination"].astype(int)
    df["label_bias"] = df["label_bias"].astype(int)

    # Normalize text fields
    df["ai_response"] = df["ai_response"].astype(str).str.strip()
    df["topic"] = df["topic"].astype(str).str.strip().fillna("general")

    df["hallucination_type"] = df["hallucination_type"].astype(str).str.lower().fillna("none")
    df["bias_type"] = df["bias_type"].astype(str).str.lower().fillna("none")

    return df


# -------------------------------------------------
# SPLIT FEATURES & LABELS
# -------------------------------------------------
def get_training_sets(df: pd.DataFrame):
    """
    Returns feature text and labels for all ML tasks.
    """

    # Combine topic + response (helps ML)
    X = df["topic"] + " : " + df["ai_response"]

    y_hallucination_flag = df["label_hallucination"]
    y_hallucination_type = df["hallucination_type"]

    y_bias_flag = df["label_bias"]
    y_bias_type = df["bias_type"]

    return (
        X,
        y_hallucination_flag,
        y_hallucination_type,
        y_bias_flag,
        y_bias_type
    )


# -------------------------------------------------
# DEBUG / SANITY CHECK
# -------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    df_clean = preprocess_dataset(df)

    print("Dataset shape after cleaning:", df_clean.shape)
    print("\nSample rows:")
    print(df_clean.head())

    (
        X,
        y_h_flag,
        y_h_type,
        y_b_flag,
        y_b_type
    ) = get_training_sets(df_clean)

    print("\nSample input text:")
    print(X.iloc[0])

    print("\nSample labels:")
    print("Hallucination flag:", y_h_flag.iloc[0])
    print("Hallucination type:", y_h_type.iloc[0])
    print("Bias flag:", y_b_flag.iloc[0])
    print("Bias type:", y_b_type.iloc[0])
