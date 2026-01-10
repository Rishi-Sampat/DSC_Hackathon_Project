from pipeline import run_pipeline

while True:
    text = input("\nEnter a statement (or 'exit'): ")
    if text.lower() == "exit":
        break

    result = run_pipeline(text)

    print("\nFINAL OUTPUT")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k}: {v}")
