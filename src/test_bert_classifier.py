from src.bert_classifier import BertEmotionClassifier

classifier = BertEmotionClassifier()

examples = [
    "I am really frustrated today.",
    "Why does this happen?",
    "I finally solved it!",
    "Nothing interesting is happening today."
]

for text in examples:

    print("=" * 70)
    print("Text:", text)
    print()

    result = classifier.predict(text)

    print("Emotion:", result["emotion"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")

    print("\nScores:")

    for emotion, score in result["scores"].items():
        print(f"{emotion:12}: {score:.4f}")