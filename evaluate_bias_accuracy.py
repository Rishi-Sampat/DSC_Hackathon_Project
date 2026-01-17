from pipeline import run_pipeline
from data_preprocessing import load_dataset

df = load_dataset()

correct = 0
total = 0

for _, row in df.iterrows():
    text = row["ai_response"]

    # Ground truth bias label
    true_bias = row["label_bias"] == 1

    # Model prediction
    result = run_pipeline(text)
    pred_bias = bool(result["bias_detected"])

    # Compare
    if pred_bias == true_bias:
        correct += 1

    total += 1

accuracy = (correct / total) * 100

print("\n==============================")
print("BIAS DETECTION ACCURACY")
print("==============================")
print(f"Correct: {correct} out of {total}")
print(f"Accuracy: {accuracy:.2f}%")
print("==============================\n")
