from src.fusion_classifier import HybridEmotionClassifier

classifier = HybridEmotionClassifier()

examples = [

    "I am really frustrated today.",

    "I finally solved the assignment!",

    "Nothing interesting is happening today.",

    "I don't understand this topic.",

    "I am excited to learn AI."

]

for text in examples:

    print("=" * 70)

    print("Text:", text)

    emotion, confidence, scores = classifier.predict(text)

    print()

    print("Final Emotion:", emotion)

    print("Confidence:", round(confidence * 100, 2), "%")

    print()

    print("Final Scores:")

    for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):

        print(f"{e:<12}: {s:.4f}")