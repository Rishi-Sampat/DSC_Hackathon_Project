from pipeline import run_pipeline
from data_preprocessing import load_dataset

def evaluate_accuracy():
    df = load_dataset()

    total = 0
    correct = 0

    for _, row in df.iterrows():
        text = row["ai_response"]

    # Ground truth: 1 and 2 both mean hallucination
        true_label = row["label_hallucination"] in [1, 2]

    # Run model
        result = run_pipeline(text)
        pred_label = bool(result["hallucination_detected"])

    # Compare
        if pred_label == true_label:
            correct += 1

        total += 1

    accuracy = correct / total
    print("\n==============================")
    print("FINAL HALLUCINATION ACCURACY")
    print("==============================")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    evaluate_accuracy()
